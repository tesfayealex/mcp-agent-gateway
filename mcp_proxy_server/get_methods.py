import asyncio
import logging
import inspect
from typing import Any, Dict, Callable

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPToolDef
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


def _create_dynamic_wrapper_function(server_wrapper: MCPClientWrapper, original_tool_name: str, original_tool_description: str, tool_def: FastMCPToolDef) -> Callable:
    """
    Creates a dynamic wrapper function with the proper signature that FastMCP expects.
    Instead of accepting arguments as a dict, we create a function with individual parameters
    that match the original tool's schema.
    """
    # Extract parameter information from the tool definition
    parameters_schema = getattr(tool_def, 'parameters', {})
    properties = parameters_schema.get('properties', {}) if parameters_schema else {}
    required_params = set(parameters_schema.get('required', [])) if parameters_schema else set()
    
    logger.debug(f"Creating wrapper for '{original_tool_name}' with schema: {parameters_schema}")
    
    # Build the function signature dynamically
    param_annotations = {}
    param_defaults = {}
    
    for param_name, param_schema in properties.items():
        # Map JSON Schema types to Python types
        param_type = param_schema.get('type', 'string')
        python_type = Any  # Default fallback
        
        if param_type == 'string':
            python_type = str
        elif param_type == 'integer':
            python_type = int
        elif param_type == 'number':
            python_type = float
        elif param_type == 'boolean':
            python_type = bool
        elif param_type == 'array':
            python_type = list
        elif param_type == 'object':
            python_type = dict
        
        # Handle optional parameters (not in required list)
        if param_name not in required_params:
            python_type = python_type | None  # Make it optional using union syntax
            param_defaults[param_name] = None
        
        param_annotations[param_name] = python_type
    
    # Create the wrapper function dynamically
    def create_wrapper():
        async def dynamic_wrapper(**kwargs) -> Any:
            # Convert kwargs back to the arguments dict that the downstream tool expects
            return await _create_proxied_tool_callable(
                server_wrapper=server_wrapper,
                original_tool_name=original_tool_name,
                arguments=kwargs
            )
        
        return dynamic_wrapper
    
    wrapper_func = create_wrapper()
    
    # Set the function signature dynamically
    # Create Parameter objects for the signature
    sig_params = []
    for param_name, param_type in param_annotations.items():
        default_value = param_defaults.get(param_name, inspect.Parameter.empty)
        param = inspect.Parameter(
            name=param_name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=default_value,
            annotation=param_type
        )
        sig_params.append(param)
    
    # Set the signature on the wrapper function
    new_signature = inspect.Signature(parameters=sig_params, return_annotation=Any)
    wrapper_func.__signature__ = new_signature
    
    # Set proper metadata for FastMCP
    wrapper_func.__name__ = f"{server_wrapper.config.name}_{original_tool_name}"
    wrapper_func.__doc__ = f"[Proxied from {server_wrapper.config.name}] {original_tool_description}"
    wrapper_func.__annotations__ = param_annotations.copy()
    wrapper_func.__annotations__['return'] = Any
    
    logger.debug(f"Created wrapper function with signature: {new_signature}")
    
    return wrapper_func


async def register_downstream_tools_on_proxy(
    proxy_mcp_instance: FastMCP,
    connection_manager: MCPConnectionManager
):
    """
    Fetches tools from all connected downstream MCP servers and registers them
    on the main proxy_mcp_instance.
    """
    logger.info("Starting registration of downstream tools on the proxy...")
    registered_tool_count = 0
    
    # Debug: Check proxy tools before registration
    logger.info("=== DEBUGGING: Checking proxy tools before downstream registration ===")
    try:
        # Check if proxy has any tools already
        if hasattr(proxy_mcp_instance, '_tools'):
            logger.info(f"Proxy instance has _tools attribute: {proxy_mcp_instance._tools}")
        if hasattr(proxy_mcp_instance, 'tools'):
            logger.info(f"Proxy instance has tools attribute: {proxy_mcp_instance.tools}")
        
        # Try to get the current tools via list_tools if available
        if hasattr(proxy_mcp_instance, 'list_tools'):
            current_tools = await proxy_mcp_instance.list_tools()
            logger.info(f"Current tools in proxy (via list_tools): {current_tools}")
        else:
            logger.info("Proxy instance doesn't have list_tools method")
            
    except Exception as e:
        logger.error(f"Error checking existing proxy tools: {e}")
    
    if not connection_manager.server_handlers:
        logger.warning("No server handlers found in connection manager. No downstream tools to register.")
        return

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

            if downstream_tools is None: # Should not happen if list_tools is well-behaved
                logger.error(f"Failed to retrieve tools from '{server_name}'; server returned None. Skipping.")
                continue
            if not isinstance(downstream_tools, list):
                logger.error(f"Tool list from '{server_name}' is not a list (type: {type(downstream_tools)}). Skipping.")
                continue
            
            logger.debug(f"Successfully fetched {len(downstream_tools)} tool definitions from '{server_name}'.")

            for tool_def in downstream_tools:
                logger.debug(f"Processing tool_def from '{server_name}': {tool_def}") # Log the raw tool_def

                if not hasattr(tool_def, 'name') or not tool_def.name:
                    logger.warning(f"Found a tool from server '{server_name}' without a name. Skipping: {tool_def}")
                    continue

                original_tool_name = tool_def.name
                prefixed_tool_name = f"{server_name}_{original_tool_name}"
                
                original_description = getattr(tool_def, 'description', "No description provided.")
                proxied_description = f"[Proxied from {server_name}] {original_description}"

                # Create a clean wrapper function with proper signature based on tool schema
                wrapper_function = _create_dynamic_wrapper_function(
                    server_wrapper=wrapper,
                    original_tool_name=original_tool_name,
                    original_tool_description=original_description,
                    tool_def=tool_def
                )

                tool_params_for_log = getattr(tool_def, 'parameters', "N/A")
                logger.debug(f"Attempting to add tool to proxy: Name='{prefixed_tool_name}', OriginalName='{original_tool_name}', Description='{proxied_description}', OriginalParams='{tool_params_for_log}'")
                
                # Debug: Check the wrapper function signature
                try:
                    sig = inspect.signature(wrapper_function)
                    logger.debug(f"Wrapper function signature for '{prefixed_tool_name}': {sig}")
                except Exception as e:
                    logger.error(f"Could not inspect signature of wrapper function: {e}")

                try:
                    logger.debug(f"About to call proxy_mcp_instance.add_tool for '{prefixed_tool_name}'...")
                    proxy_mcp_instance.add_tool(
                        fn=wrapper_function,
                        name=prefixed_tool_name,
                        description=proxied_description,
                    )
                    logger.info(f"Successfully ADDED tool '{prefixed_tool_name}' (from '{server_name}.{original_tool_name}') to proxy instance.")
                    registered_tool_count += 1
                    
                    # Debug: Check if tool was actually added
                    try:
                        if hasattr(proxy_mcp_instance, '_tools'):
                            logger.debug(f"After adding '{prefixed_tool_name}', proxy _tools keys: {list(proxy_mcp_instance._tools.keys()) if proxy_mcp_instance._tools else 'None'}")
                        if hasattr(proxy_mcp_instance, 'tools'):
                            logger.debug(f"After adding '{prefixed_tool_name}', proxy tools keys: {list(proxy_mcp_instance.tools.keys()) if proxy_mcp_instance.tools else 'None'}")
                    except Exception as e:
                        logger.error(f"Error checking proxy tools after adding '{prefixed_tool_name}': {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to ADD tool '{prefixed_tool_name}' to proxy instance: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error processing tools for server '{server_name}': {e}", exc_info=True)
    
    # Debug: Check proxy tools after registration
    logger.info("=== DEBUGGING: Checking proxy tools after downstream registration ===")
    try:
        if hasattr(proxy_mcp_instance, '_tools'):
            logger.info(f"Final proxy _tools: {proxy_mcp_instance._tools}")
        if hasattr(proxy_mcp_instance, 'tools'):
            logger.info(f"Final proxy tools: {proxy_mcp_instance.tools}")
            
        # Try to get the final tools via list_tools if available
        if hasattr(proxy_mcp_instance, 'list_tools'):
            final_tools = await proxy_mcp_instance.list_tools()
            logger.info(f"Final tools in proxy (via list_tools): {final_tools}")
        else:
            logger.info("Proxy instance doesn't have list_tools method")
            
    except Exception as e:
        logger.error(f"Error checking final proxy tools: {e}")
    
    logger.info(f"Completed downstream tool registration. Total tools registered: {registered_tool_count}") 