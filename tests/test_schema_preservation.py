import asyncio
import json
import logging
from typing import Any, Dict

from fastmcp.client import Client
from fastmcp.client.transports import SSETransport

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_schema_preservation():
    """Test whether proxy server preserves schema information"""
    
    # Connect to proxy server
    proxy_url = "http://localhost:8001/sse"
    logger.info(f"Connecting to proxy at {proxy_url}")
    
    async with Client(SSETransport(proxy_url)) as proxy_client:
        logger.info("Connected to proxy, listing tools...")
        
        # Get all tools - fix: it returns a list directly
        tools = await proxy_client.list_tools()
        logger.info(f"Found {len(tools)} tools")
        
        # Print schema information for first few tools
        print(f"=== Found {len(tools)} tools ===")
        for i, tool in enumerate(tools[:3]):
            print(f"\n=== Tool {i+1}: {tool.name} ===")
            print(f"Description: {tool.description}")
            print(f"Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
        
        # Let's also check what managed servers are available
        logger.info("Testing list_managed_servers to see downstream tool data...")
        
        try:
            managed_servers_result = await proxy_client.call_tool("list_managed_servers", {})
            logger.info(f"Managed servers result: {managed_servers_result.content}")
            
            # Try to access the proxy server internals by importing
            # This will only work if running from the same process/environment
            try:
                from mcp_proxy_server.proxy_server import proxy_server
                logger.info("Checking proxy server internal state...")
                
                # Check if we can access downstream tools
                if hasattr(proxy_server, 'downstream_tools'):
                    logger.info(f"Downstream tools count: {len(proxy_server.downstream_tools)}")
                    for tool_name, tool_data in list(proxy_server.downstream_tools.items())[:3]:
                        logger.info(f"Tool: {tool_name}")
                        logger.info(f"Tool data type: {type(tool_data)}")
                        if hasattr(tool_data, 'parameters'):
                            logger.info(f"Parameters: {tool_data.parameters}")
                        if hasattr(tool_data, 'inputSchema'):
                            logger.info(f"InputSchema: {tool_data.inputSchema}")
                        logger.info("---")
                        
            except ImportError as e:
                logger.info(f"Cannot access proxy server internals from client: {e}")
            
        except Exception as e:
            logger.error(f"Error checking downstream data: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_schema_preservation()) 