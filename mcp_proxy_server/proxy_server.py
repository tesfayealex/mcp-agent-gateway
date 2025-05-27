import asyncio
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager # Added import

from fastmcp import FastMCP
# Removed: from fastmcp.server import ToolResponse, ToolContext
# from fastmcp.fast_api_app import FastAPIMethod # For type hinting if needed for app events

# Adjust sys.path to allow importing from mcp_manager if mcp-agent-gateway is the project root
# This is often needed if running proxy_server.py directly during development.
# For production, consider packaging your project properly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from mcp_manager.connection_manager import MCPConnectionManager
    from mcp_manager.config_loader import load_configs, ServerConfig
    from mcp_proxy_server.models import (
        ManagedServerInfo,
        DownstreamToolSchema,
        ToolParameterSchema,
        ToolCallResult,
        ListDownstreamToolsResponse,
        ErrorResponse
    )
except ImportError as e:
    print(f"Error importing modules. Ensure mcp_manager is in PYTHONPATH or installed. Details: {e}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
# This will hold the initialized MCPConnectionManager instance
mcp_conn_manager_instance: Optional[MCPConnectionManager] = None
proxy_mcp_config: Optional[List[ServerConfig]] = None

# --- Lifespan manager for FastMCP ---
@asynccontextmanager
async def app_lifespan(fast_mcp_server: FastMCP): # server instance is passed by FastMCP
    global mcp_conn_manager_instance, proxy_mcp_config
    logger.info("MCP Gateway Proxy starting up via lifespan...")
    try:
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"config.json not found at {config_path}")

        loaded_server_configs = load_configs(config_path)
        
        # Initialize manager instance
        mcp_conn_manager_instance = MCPConnectionManager(server_configs=loaded_server_configs)
        await mcp_conn_manager_instance.connect_all_servers()
        asyncio.create_task(mcp_conn_manager_instance.start_monitoring())
        logger.info("MCPConnectionManager initialized and server connections initiated.")
        
        # Set the global config list after successful loading and manager init
        proxy_mcp_config = loaded_server_configs
        
    except Exception as e:
        logger.error(f"Error during startup via lifespan: {e}", exc_info=True)
        # Re-raise to prevent server from starting in a bad state
        raise

    yield # Server runs here

    logger.info("MCP Gateway Proxy shutting down via lifespan...")
    if mcp_conn_manager_instance:
        await mcp_conn_manager_instance.stop_monitoring()
        await mcp_conn_manager_instance.disconnect_all_servers()
        logger.info("MCPConnectionManager resources released.")
    else:
        logger.info("MCPConnectionManager was not initialized, no resources to release during shutdown.")

# --- FastMCP Server Instance ---
proxy_mcp = FastMCP(
    name="MCPGatewayProxy",
    description="A proxy server to manage and interact with multiple downstream MCP servers.",
    lifespan=app_lifespan,  # Use the new lifespan manager
    streamable_http_path="/proxy"  # Set the base path for MCP HTTP endpoints
)

# --- Helper to get the connection manager ---
async def get_mcp_conn_manager() -> MCPConnectionManager:
    if mcp_conn_manager_instance is None:
        logger.error("MCPConnectionManager not initialized. This should happen at startup via lifespan.")
        raise RuntimeError("MCPConnectionManager not initialized.")
    return mcp_conn_manager_instance

# --- MCP Tools for the Proxy Server (Revised) ---

# Helper to return errors in a consistent dictionary format
def create_error_dict(message: str, status_code: int = 500) -> Dict[str, Any]:
    logger.error(f"Tool error: {message} (status_code: {status_code})")
    return {"mcp_proxy_error": True, "detail": message, "status_code": status_code}

@proxy_mcp.tool()
async def list_managed_servers() -> List[ManagedServerInfo] | Dict[str, Any]:
    """Lists all downstream MCP servers managed by this proxy and their status."""
    try:
        manager = await get_mcp_conn_manager()
        infos: List[ManagedServerInfo] = []
        # Ensure proxy_mcp_config (the list of ServerConfig) is available
        current_configs = proxy_mcp_config
        if current_configs is None:
            # This might happen if startup failed before proxy_mcp_config was set
            logger.warning("proxy_mcp_config is None in list_managed_servers. Startup might have failed.")
            # Fallback to an empty list or error, depending on desired behavior
            # For now, proceed, but config_enabled might be inaccurate
            current_configs = []


        for server_name, wrapper in manager.server_handlers.items():
            config_enabled = True 
            original_server_config = next((s for s in current_configs if s.name == server_name), None)
            if original_server_config:
                config_enabled = original_server_config.enabled
            else:
                # If not found in current_configs (e.g. config reloaded elsewhere, or startup issue)
                # We might assume it was enabled to be in server_handlers, or stick to a default.
                # For safety, if it's in server_handlers, it was enabled at some point.
                # But if proxy_mcp_config is the source of truth for *initial* enabled state, this is tricky.
                # The MCPConnectionManager only adds enabled servers.
                pass # config_enabled remains true as per current logic if not found specifically

            infos.append(
                ManagedServerInfo(
                    name=wrapper.config.name,
                    status=wrapper.status,
                    connection_type=wrapper.config.connection_type,
                    config_enabled=config_enabled
                )
            )
        return infos
    except RuntimeError as e: 
        return create_error_dict(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in list_managed_servers: {e}", exc_info=True)
        return create_error_dict("An unexpected error occurred while listing managed servers.")


@proxy_mcp.tool()
async def get_server_tools(server_name: str) -> ListDownstreamToolsResponse | Dict[str, Any]:
    """
    Retrieves the list of tools available on a specific downstream MCP server.
    Args:
        server_name: The name of the downstream MCP server (as defined in the config).
    """
    try:
        manager = await get_mcp_conn_manager()
        wrapper = manager.get_server_handler(server_name)
        if not wrapper:
            return create_error_dict(f"Server '{server_name}' not found.", status_code=404)
        if not wrapper.config.enabled:
             return create_error_dict(f"Server '{server_name}' is configured but disabled.", status_code=403) 

        if wrapper.status != "connected":
            return create_error_dict(f"Server '{server_name}' is not connected. Current status: {wrapper.status}.", status_code=503)

        # downstream_tools_raw from MCPClientWrapper.list_tools() should be List[fastmcp.protocol.Tool]
        downstream_tools_raw = await wrapper.list_tools() 

        if downstream_tools_raw is None: 
             return create_error_dict(f"Failed to retrieve tools from '{server_name}', server returned no data.", status_code=502)
        
        processed_tools: List[DownstreamToolSchema] = []
        if isinstance(downstream_tools_raw, list):
            for tool_object in downstream_tools_raw: # tool_object is fastmcp.protocol.Tool
                params_schema: List[ToolParameterSchema] = []
                
                # Access attributes directly from the Pydantic model (fastmcp.protocol.Tool)
                # The 'parameters' attribute of a fastmcp.protocol.Tool object is a list of fastmcp.protocol.ToolParameter objects
                tool_parameters = tool_object.parameters if hasattr(tool_object, 'parameters') and tool_object.parameters else []

                if tool_parameters:
                    for p_obj in tool_parameters: # p_obj is fastmcp.protocol.ToolParameter
                        params_schema.append(
                            ToolParameterSchema(
                                name=p_obj.name if hasattr(p_obj, 'name') else None,
                                type_hint=str(p_obj.type_hint) if hasattr(p_obj, 'type_hint') else 'Any',
                                required=p_obj.required if hasattr(p_obj, 'required') else False,
                                description=p_obj.description if hasattr(p_obj, 'description') else None,
                                default=p_obj.default if hasattr(p_obj, 'default') else None # Handle if default is not present
                            )
                        )
                
                tool_name = tool_object.name if hasattr(tool_object, 'name') else "Unknown Tool"
                tool_description = tool_object.description if hasattr(tool_object, 'description') else "No description"

                processed_tools.append(
                    DownstreamToolSchema(
                        name=tool_name,
                        description=tool_description,
                        parameters=params_schema
                    )
                )
        else:
            logger.warning(f"Received unexpected tool list format from {server_name}: {type(downstream_tools_raw)}. Expected a list.")
            return create_error_dict(f"Received unexpected tool list format from '{server_name}'. Expected a list.", status_code=502)

        return ListDownstreamToolsResponse(tools=processed_tools)

    except RuntimeError as e: 
        return create_error_dict(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_server_tools for '{server_name}': {e}", exc_info=True)
        return create_error_dict(f"An unexpected error occurred while getting tools for '{server_name}'.")


@proxy_mcp.tool()
async def call_server_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any]
) -> ToolCallResult | Dict[str, Any]:
    """
    Calls a specific tool on a downstream MCP server.
    Args:
        server_name: The name of the downstream MCP server.
        tool_name: The name of the tool to call on the downstream server.
        arguments: A dictionary of arguments to pass to the downstream tool.
    """
    try:
        manager = await get_mcp_conn_manager()
        wrapper = manager.get_server_handler(server_name)

        if not wrapper:
            return create_error_dict(f"Server '{server_name}' not found.", status_code=404)
        if not wrapper.config.enabled:
            return create_error_dict(f"Server '{server_name}' is configured but disabled.", status_code=403)

        if not await wrapper.ensure_connected():
             return create_error_dict(f"Failed to connect to server '{server_name}'. Current status: {wrapper.status}.", status_code=503)

        logger.info(f"Proxying call to '{tool_name}' on server '{server_name}' with args: {arguments}")
        result = await wrapper.call_tool(tool_name=tool_name, params=arguments)
        print(result)
        if wrapper.status == "error" and result is None: 
            return ToolCallResult(success=False, error_message=f"Execution of '{tool_name}' on '{server_name}' failed. Server status: error.")
        
        return ToolCallResult(success=True, result=result)

    except RuntimeError as e: 
        return create_error_dict(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in call_server_tool for '{server_name}/{tool_name}': {e}", exc_info=True)
        return create_error_dict(f"An unexpected error occurred while calling tool '{tool_name}' on '{server_name}'.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # uvicorn is imported locally as it's specific to HTTP execution
    # FastMCP handles uvicorn interaction internally when using proxy_mcp.run()

    config_file_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(config_file_path):
        logger.error(f"CRITICAL: config.json not found at {config_file_path}. The proxy server cannot start without configuration.")
        logger.error("Please create a config.json file in the project root (mcp-agent-gateway).")
        sys.exit(1)

    logger.info(f"Attempting to start MCP Gateway Proxy on port 8001...")
    logger.info(f"Project root (for config.json): {PROJECT_ROOT}")
    logger.info(f"FastMCP HTTP base path configured to: {proxy_mcp.settings.streamable_http_path}")

    # Use FastMCP's built-in run method for uvicorn
    # This will use the lifespan manager correctly.
    try:
        proxy_mcp.run(
            transport="streamable-http", # Default transport for uvicorn
            host="0.0.0.0",
            port=8001,
            log_level="info" # uvicorn log level
        )
    except Exception as e:
        logger.critical(f"Failed to start MCP Gateway Proxy: {e}", exc_info=True)
        sys.exit(1) 