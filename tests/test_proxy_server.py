import asyncio
import logging
import json # Added for parsing JSON
from typing import Any, Dict, List, Optional

from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent # Added for type checking

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestProxyClient")

PROXY_SERVER_URL = "http://localhost:8001/proxy" # Default URL for the proxy

async def main():
    logger.info(f"Attempting to connect to MCP Gateway Proxy at {PROXY_SERVER_URL}")
    
    transport = StreamableHttpTransport(url=PROXY_SERVER_URL)
    client = Client(transport=transport)

    try:
        async with client: # Use the Client instance as the async context manager
            logger.info("Client context entered. MCP Client should be active.")

            # 1. Test list_managed_servers
            logger.info("\n--- Testing 'list_managed_servers' ---")
            managed_servers_result = await client.call_tool("list_managed_servers")
            
            managed_servers: List[Dict[str, Any]] = [] # Initialize to empty list

            if not managed_servers_result:
                logger.error("'list_managed_servers' returned None or empty.")
                # No return here, managed_servers remains empty, subsequent checks will handle it
            
            elif isinstance(managed_servers_result, dict) and managed_servers_result.get("mcp_proxy_error"):
                logger.error(f"Error from 'list_managed_servers': {managed_servers_result.get('detail')}")
            
            elif isinstance(managed_servers_result, list) and len(managed_servers_result) > 0 and isinstance(managed_servers_result[0], TextContent):
                # Handle the case where the result is a list containing TextContent with JSON
                try:
                    logger.info("'list_managed_servers' returned TextContent, attempting to parse JSON.")
                    # Assuming the actual list of servers is in the text of the first TextContent element
                    # This might need adjustment if multiple TextContent objects or other structures are possible
                    json_text = managed_servers_result[0].text
                    parsed_servers = json.loads(json_text)
                    if isinstance(parsed_servers, list):
                        managed_servers = parsed_servers
                        logger.info(f"Successfully parsed managed servers: {managed_servers}")
                    else:
                        logger.error(f"Parsed JSON from TextContent is not a list: {type(parsed_servers)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from TextContent: {e}")
                except Exception as e:
                    logger.error(f"Error processing TextContent from 'list_managed_servers': {e}")
            
            elif isinstance(managed_servers_result, list):
                # If it's a list, but not of TextContent, assume it's already the list of dicts
                # This might need more robust checking if other list contents are possible
                all_dicts = True
                for item in managed_servers_result:
                    if not isinstance(item, dict):
                        all_dicts = False
                        break
                if all_dicts:
                    managed_servers = managed_servers_result
                    logger.info(f"Managed servers (list of dicts): {managed_servers}")
                else:
                    logger.error(f"'list_managed_servers' returned a list with non-dictionary items: {managed_servers_result}")
            else:
                logger.error(f"'list_managed_servers' returned an unexpected type: {type(managed_servers_result)}")

            if not managed_servers:
                logger.warning("No managed servers parsed or an error occurred. Skipping further tests that depend on a server.")
                return
            
            # Find the first connected and enabled server for further tests
            target_server_name = None
            for server_info in managed_servers:
                if isinstance(server_info, dict): # Ensure server_info is a dict before calling .get
                    if server_info.get("status") == "connected" and server_info.get("config_enabled"):
                        target_server_name = server_info.get("name")
                        logger.info(f"Using server '{target_server_name}' for further tests.")
                        break
                else:
                    logger.warning(f"Skipping non-dictionary item in managed_servers list: {server_info}")
            
            if not target_server_name:
                logger.warning("No connected and enabled server found. Skipping tests for get_server_tools and call_server_tool.")
                return

            # 2. Test get_server_tools
            logger.info(f"\n--- Testing 'get_server_tools' for server: {target_server_name} ---")
            server_tools_result_raw = await client.call_tool(
                "get_server_tools",
                arguments={"server_name": target_server_name}
            )

            server_tools_data: Optional[Dict[str, Any]] = None
            target_tool_name = None
            target_tool_params = None

            if not server_tools_result_raw:
                logger.error(f"'get_server_tools' for '{target_server_name}' returned None or empty.")
            elif isinstance(server_tools_result_raw, list) and len(server_tools_result_raw) > 0 and isinstance(server_tools_result_raw[0], TextContent):
                try:
                    logger.info(f"'get_server_tools' for '{target_server_name}' returned TextContent, attempting to parse JSON.")
                    json_text = server_tools_result_raw[0].text
                    parsed_data = json.loads(json_text)
                    if isinstance(parsed_data, dict):
                        server_tools_data = parsed_data
                    else:
                        logger.error(f"Parsed JSON from TextContent for 'get_server_tools' is not a dict: {type(parsed_data)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from TextContent for 'get_server_tools': {e}")
            elif isinstance(server_tools_result_raw, dict):
                server_tools_data = server_tools_result_raw
            else:
                logger.error(f"'get_server_tools' for '{target_server_name}' returned an unexpected raw type: {type(server_tools_result_raw)}, content: {server_tools_result_raw}")

            if server_tools_data:
                if server_tools_data.get("mcp_proxy_error"):
                    logger.error(f"Error from 'get_server_tools' for '{target_server_name}': {server_tools_data.get('detail')}")
                elif "tools" in server_tools_data:
                    tools_list = server_tools_data.get("tools", [])
                    logger.info(f"Tools on server '{target_server_name}': {tools_list}")
                    if tools_list:
                        for tool_info in tools_list:
                            if tool_info.get("name") == "get_me":
                                target_tool_name = "get_me"
                                target_tool_params = {}
                                break
                        if not target_tool_name and tools_list:
                            target_tool_name = tools_list[0].get("name")
                            target_tool_params = {}
                            logger.info(f"'get_me' not found, selected first tool '{target_tool_name}' for testing call_server_tool.")
                        if not target_tool_name:
                            logger.warning(f"No tools found on server '{target_server_name}' to test with 'call_server_tool'.")
                    else:
                        logger.warning(f"No tools listed for server '{target_server_name}'.")
                else:
                    logger.error(f"'get_server_tools' processed data for '{target_server_name}' has unexpected structure: {server_tools_data}")
            elif not server_tools_result_raw:
                pass
            else:
                logger.error(f"Could not process result from 'get_server_tools' for {target_server_name}.")

            if not target_tool_name:
                logger.warning(f"Could not determine a target tool on '{target_server_name}'. Skipping 'call_server_tool' test.")
                return

            # 3. Test call_server_tool
            logger.info(f"\n--- Testing 'call_server_tool': server='{target_server_name}', tool='{target_tool_name}', args={target_tool_params} ---")
            tool_call_response_raw = await client.call_tool(
                "call_server_tool",
                arguments={
                    "server_name": target_server_name,
                    "tool_name": target_tool_name,
                    "arguments": target_tool_params
                }
            )
            
            tool_call_response_data: Optional[Dict[str, Any]] = None

            if not tool_call_response_raw:
                logger.error(f"'call_server_tool' for '{target_tool_name}' on '{target_server_name}' returned None or empty.")
            elif isinstance(tool_call_response_raw, list) and len(tool_call_response_raw) > 0 and isinstance(tool_call_response_raw[0], TextContent):
                try:
                    logger.info(f"'call_server_tool' for '{target_tool_name}' returned TextContent, attempting to parse JSON.")
                    json_text = tool_call_response_raw[0].text
                    parsed_data = json.loads(json_text)
                    if isinstance(parsed_data, dict):
                        tool_call_response_data = parsed_data
                    else:
                        logger.error(f"Parsed JSON from TextContent for 'call_server_tool' is not a dict: {type(parsed_data)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from TextContent for 'call_server_tool': {e}")
            elif isinstance(tool_call_response_raw, dict):
                tool_call_response_data = tool_call_response_raw
            else:
                logger.error(f"'call_server_tool' for '{target_tool_name}' on '{target_server_name}' returned an unexpected raw type: {type(tool_call_response_raw)}, content: {tool_call_response_raw}")

            if tool_call_response_data:
                if tool_call_response_data.get("mcp_proxy_error"):
                    logger.error(f"Error from 'call_server_tool': {tool_call_response_data.get('detail')}")
                else:
                    logger.info(f"Result from 'call_server_tool' for '{target_tool_name}' on '{target_server_name}':")
                    logger.info(f"  Success: {tool_call_response_data.get('success')}")
                    if tool_call_response_data.get('success'):
                        logger.info(f"  Result: {tool_call_response_data.get('result')}")
                    else:
                        logger.error(f"  Error Message: {tool_call_response_data.get('error_message')}")
            elif not tool_call_response_raw:
                pass
            else:
                logger.error(f"Could not process result from 'call_server_tool' for {target_server_name}/{target_tool_name}.")

    except ConnectionRefusedError:
        logger.error(f"Connection refused. Ensure the MCP Gateway Proxy server is running at {PROXY_SERVER_URL}.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    logger.info("Test script finished.")

if __name__ == "__main__":
    logger.info("Starting MCP Gateway Proxy test script...")
    logger.info("IMPORTANT: Ensure the 'proxy_server.py' is running before starting this test.")
    asyncio.run(main()) 