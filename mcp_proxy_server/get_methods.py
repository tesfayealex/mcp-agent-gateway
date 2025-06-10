import asyncio
import logging
import inspect
from typing import Any, Dict, Callable, Union, Optional

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPToolDef
from fastmcp.tools.tool import Tool as FastMCPTool  # Import the actual Tool class for direct creation
# import fastmcp.tools # Removed if not used elsewhere
# import fastmcp.utilities # Removed if not used elsewhere

from mcp_manager.connection_manager import MCPConnectionManager
from mcp_manager.server_handler import MCPClientWrapper

import fastmcp

# fastmcp.tools. # This was a commented out line, will be removed by omission if not present in original
logger = logging.getLogger(__name__)

async def _create_proxied_tool_callable(
    server_wrapper: MCPClientWrapper, 
    original_tool_name: str,
    arguments: Dict[str, Any] # This will be populated by FastMCP when the tool is called
) -> Any:
    """
    This is the actual function that gets called when a proxied tool is invoked.
    It uses the captured server_wrapper and original_tool_name to call the downstream tool.
    """
    logger.info(f"Proxying call to '{original_tool_name}' on server '{server_wrapper.config.name}' with args: {arguments}")
    
    if not await server_wrapper.ensure_connected():
        error_msg = f"Failed to connect to server '{server_wrapper.config.name}' for tool '{original_tool_name}'. Current status: {server_wrapper.status}."
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    try:
        result = await server_wrapper.call_tool(tool_name=original_tool_name, params=arguments)
        
        if server_wrapper.status == "error" and result is None:
            logger.error(f"Execution of '{original_tool_name}' on '{server_wrapper.config.name}' failed. Server status: error.")
            raise Exception(f"Tool execution failed on downstream server '{server_wrapper.config.name}'.")

        return result
    except Exception as e:
        logger.error(f"Error during proxied call to '{original_tool_name}' on '{server_wrapper.config.name}': {e}", exc_info=True)
        raise


def _create_dynamic_wrapper_function(tool_def, downstream_client, server_name):
    """
    Create a dynamic wrapper function that preserves the original tool's parameter schema.
    Instead of letting FastMCP generate schema from function signature, we create Tool objects
    directly with the preserved original schema.
    """
    # Extract the original parameter schema from the tool definition
    original_schema = getattr(tool_def, 'inputSchema', None) or getattr(tool_def, 'parameters', {})
    
    logger.debug(f"Creating wrapper for tool {tool_def.name} with schema: {original_schema}")
    
    # Build function signature based on original schema
    sig_params = []
    annotations = {}
    
    if original_schema and isinstance(original_schema, dict):
        properties = original_schema.get('properties', {})
        required = original_schema.get('required', [])
        
        # Sort parameters: required first, then optional
        # This prevents "non-default argument follows default argument" error
        required_params = []
        optional_params = []
        
        for param_name, param_info in properties.items():
            # Map JSON schema types to Python types
            param_type = str  # Default to str
            if param_info.get('type') == 'integer':
                param_type = int
            elif param_info.get('type') == 'number':
                param_type = float
            elif param_info.get('type') == 'boolean':
                param_type = bool
            elif param_info.get('type') == 'array':
                param_type = list
            elif param_info.get('type') == 'object':
                param_type = dict
            
            # Create parameter with default if not required
            if param_name in required:
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_type)
                required_params.append(param)
            else:
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                        annotation=param_type, default=None)
                optional_params.append(param)
            
            annotations[param_name] = param_type
        
        # Combine required parameters first, then optional
        sig_params = required_params + optional_params
    
    # Create the dynamic wrapper function
    async def wrapper_func(**kwargs):
        logger.debug(f"Calling tool {tool_def.name} with args: {kwargs}")
        try:
            result = await downstream_client.call_tool(tool_def.name, kwargs)
            logger.debug(f"Tool {tool_def.name} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_def.name}: {e}")
            raise
    
    # Set function metadata
    wrapper_func.__name__ = tool_def.name
    wrapper_func.__doc__ = tool_def.description or f"Proxied tool {tool_def.name} from {server_name}"
    wrapper_func.__annotations__ = annotations
    wrapper_func.__signature__ = inspect.Signature(sig_params)
    
    # Store the original schema for direct Tool creation
    wrapper_func.__schema__ = original_schema
    
    logger.debug(f"Created wrapper function for {tool_def.name} with preserved schema")
    
    return wrapper_func


def _register_tools_with_fastmcp(app, tools_by_server):
    """
    Register tools with FastMCP using direct Tool creation to preserve schemas.
    """
    from fastmcp.tools import Tool
    
    for server_name, tools in tools_by_server.items():
        logger.info(f"Registering {len(tools)} tools from {server_name}")
        
        for tool_func in tools:
            try:
                # Get the preserved schema
                original_schema = getattr(tool_func, '__schema__', {})
                
                # Create Tool directly with preserved schema
                tool = Tool(
                    fn=tool_func,
                    name=tool_func.__name__,
                    description=tool_func.__doc__ or f"Proxied tool from {server_name}",
                    parameters=original_schema,
                    tags=set(),
                    annotations=None,
                    serializer=None
                )
                
                # Add the tool directly to the tool manager
                app._tool_manager.add_tool(tool)
                
                logger.debug(f"Registered tool {tool_func.__name__} with schema: {original_schema}")
                
            except Exception as e:
                logger.error(f"Failed to register tool {tool_func.__name__}: {e}")
                continue


async def register_downstream_tools_on_proxy(
    proxy_mcp_instance: FastMCP,
    connection_manager: MCPConnectionManager
):
    """
    Fetches tools from all connected downstream MCP servers and registers them
    on the main proxy_mcp_instance using direct Tool creation to preserve schemas.
    """
    logger.info("Starting registration of downstream tools on the proxy...")
    
    if not connection_manager.server_handlers:
        logger.warning("No server handlers found in connection manager. No downstream tools to register.")
        return

    # Collect all tools by server
    tools_by_server = {}
    
    for server_name, wrapper in connection_manager.server_handlers.items():
        if not wrapper.config.enabled:
            logger.debug(f"Skipping server '{server_name}': disabled in config.")
            continue

        if wrapper.status != "connected":
            logger.warning(f"Skipping server '{server_name}' for tool registration: not connected (status: {wrapper.status}). Tools will not be available until it connects.")
            continue

        logger.info(f"Fetching tools from downstream server: '{server_name}'")
        try:
            # list_tools() should return List[fastmcp.tools.Tool]
            downstream_tools: list[FastMCPToolDef] = await wrapper.list_tools()

            if downstream_tools is None:
                logger.error(f"Failed to retrieve tools from '{server_name}'; server returned None. Skipping.")
                continue
            if not isinstance(downstream_tools, list):
                logger.error(f"Tool list from '{server_name}' is not a list (type: {type(downstream_tools)}). Skipping.")
                continue
            
            logger.debug(f"Successfully fetched {len(downstream_tools)} tool definitions from '{server_name}'.")

            server_tools = []
            for tool_def in downstream_tools:
                logger.debug(f"Processing tool_def from '{server_name}': {tool_def}")

                if not hasattr(tool_def, 'name') or not tool_def.name:
                    logger.warning(f"Found a tool from server '{server_name}' without a name. Skipping: {tool_def}")
                    continue

                original_tool_name = tool_def.name
                prefixed_tool_name = f"{server_name}_{original_tool_name}"
                
                original_description = getattr(tool_def, 'description', "No description provided.")
                proxied_description = f"[Proxied from {server_name}] {original_description}"

                # Create wrapper function with preserved schema
                wrapper_function = _create_dynamic_wrapper_function(
                    tool_def=tool_def,
                    downstream_client=wrapper,
                    server_name=server_name
                )
                
                # Update the function name to include server prefix
                wrapper_function.__name__ = prefixed_tool_name
                wrapper_function.__doc__ = proxied_description
                
                server_tools.append(wrapper_function)
                logger.debug(f"Created wrapper function for '{prefixed_tool_name}' with preserved schema")

            tools_by_server[server_name] = server_tools

        except Exception as e:
            logger.error(f"Error processing tools for server '{server_name}': {e}", exc_info=True)
    
    # Register all tools with FastMCP using direct Tool creation
    _register_tools_with_fastmcp(proxy_mcp_instance, tools_by_server)
    
    # Count total registered tools
    total_tools = sum(len(tools) for tools in tools_by_server.values())
    logger.info(f"Completed downstream tool registration. Total tools registered: {total_tools}")
    
    # Debug: Check final tools
    try:
        final_tools = await proxy_mcp_instance.list_tools()
        logger.info(f"Final tools in proxy: {len(final_tools)} tools")
        # Log first few with their schemas
        for i, tool in enumerate(final_tools[:3]):
            logger.info(f"Final tool {i+1}: {tool.name}")
            logger.info(f"  Input Schema: {getattr(tool, 'inputSchema', 'N/A')}")
    except Exception as e:
        logger.error(f"Error checking final proxy tools: {e}") 