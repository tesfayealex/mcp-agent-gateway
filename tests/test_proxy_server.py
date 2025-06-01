import asyncio
import logging
import json
import pytest
import pytest_asyncio
import aiohttp
import time
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastmcp.client import Client
from fastmcp.client.transports import SSETransport
from mcp.types import TextContent

# Configure pytest-asyncio to avoid teardown errors
pytest_asyncio.asyncio_default_fixture_loop_scope = "function"

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestProxyClient")

# Test run summary logger
class TestRunSummaryLogger:
    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = []
        self.logs = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        logger.info(message)
        
    def add_test_result(self, test_name: str, status: str, duration: float, details: str = ""):
        self.test_results.append({
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "details": details
        })
        
    def write_summary(self):
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Ensure docs directory exists
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        summary_path = os.path.join(docs_dir, "test_run_summary.md")
        
        with open(summary_path, "w") as f:
            f.write(f"# Test Run Summary\n\n")
            f.write(f"**Run Date:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Duration:** {total_duration:.2f} seconds\n")
            f.write(f"**Total Tests:** {len(self.test_results)}\n")
            
            passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
            failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
            skipped = sum(1 for r in self.test_results if r["status"] == "SKIPPED")
            
            f.write(f"**Passed:** {passed} | **Failed:** {failed} | **Skipped:** {skipped}\n\n")
            
            f.write("## Test Results\n\n")
            for result in self.test_results:
                status_emoji = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⏭️"
                f.write(f"### {status_emoji} {result['test_name']}\n")
                f.write(f"- **Status:** {result['status']}\n")
                f.write(f"- **Duration:** {result['duration']:.2f}s\n")
                if result["details"]:
                    f.write(f"- **Details:** {result['details']}\n")
                f.write("\n")
            
            f.write("## Test Execution Logs\n\n")
            f.write("```\n")
            for log_entry in self.logs:
                f.write(f"{log_entry}\n")
            f.write("```\n")
            
        logger.info(f"Test run summary written to {summary_path}")

# Global test summary logger
test_summary = TestRunSummaryLogger()

PROXY_SERVER_URL = "http://localhost:8001/sse"

# Alternative HTTP-based client for testing
@pytest_asyncio.fixture
async def http_proxy_client():
    """Simple HTTP client that bypasses FastMCP client teardown issues."""
    async with aiohttp.ClientSession() as session:
        class SimpleHTTPProxyClient:
            def __init__(self, session, base_url):
                self.session = session
                self.base_url = base_url.rstrip('/')
            
            async def list_tools(self):
                async with self.session.post(f"{self.base_url}/mcp/tools/list", json={}) as response:
                    data = await response.json()
                    return data.get("tools", [])
            
            async def call_tool(self, name: str, arguments: dict = None):
                if arguments is None:
                    arguments = {}
                payload = {"name": name, "arguments": arguments}
                async with self.session.post(f"{self.base_url}/mcp/tools/call", json=payload) as response:
                    return await response.json()
        
        yield SimpleHTTPProxyClient(session, PROXY_SERVER_URL)

# Shared client fixture for all tests
@pytest_asyncio.fixture
async def proxy_client():
    """Fixture that provides a connected MCP proxy client."""
    logger.info(f"Setting up proxy client connection to {PROXY_SERVER_URL}")
    
    transport = SSETransport(url=PROXY_SERVER_URL)
    client = Client(transport=transport)

    try:
        async with client:
            logger.info("Client context entered. MCP Client should be active.")
            yield client
    except RuntimeError as e:
        if "cancel scope" in str(e):
            # Suppress known teardown errors that don't affect functionality
            logger.debug(f"Suppressing expected teardown error: {e}")
        else:
            raise
    finally:
        logger.info("Client context exited.")

# Helper function to parse tool call results
def parse_tool_result(result) -> Any:
    """Helper function to parse tool call results, handling TextContent and JSON parsing."""
    if result is None:
        return None
    
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], TextContent):
        try:
            json_text = result[0].text
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Return raw text if not JSON
            return result[0].text
    
    return result

@pytest.mark.asyncio
async def test_proxy_tools_visibility(proxy_client):
    """Test that all tools (proxy native + downstream) are visible via list_tools."""
    test_name = "test_proxy_tools_visibility"
    start_time = time.time()
    
    try:
        test_summary.log(f"=== Starting {test_name} ===")
        
        # List all tools from the proxy
        all_proxy_tools = await proxy_client.list_tools()
        test_summary.log(f"All tools reported by proxy: {[tool.name for tool in all_proxy_tools if hasattr(tool, 'name')]}")
        
        # Check for expected tool categories
        tool_names = [tool.name for tool in all_proxy_tools if hasattr(tool, 'name')]
        
        # Native proxy tools
        expected_native_tools = ["list_managed_servers", "get_server_tools"]
        
        # Expected downstream tools (assuming GitHub and filesystem servers)
        expected_github_tools = ["GITHUB_get_me"]  # Adjust based on your config
        expected_filesystem_tools = ["filesystem_list_directory"]
        
        # Check native tools
        for tool_name in expected_native_tools:
            if tool_name in tool_names:
                test_summary.log(f"✓ Found expected native tool: {tool_name}")
            else:
                test_summary.log(f"✗ Missing expected native tool: {tool_name}", "WARNING")
        
        # Check downstream tools
        for tool_name in expected_github_tools:
            if tool_name in tool_names:
                test_summary.log(f"✓ Found expected GitHub tool: {tool_name}")
            else:
                test_summary.log(f"✗ Missing expected GitHub tool: {tool_name}", "WARNING")
        
        for tool_name in expected_filesystem_tools:
            if tool_name in tool_names:
                test_summary.log(f"✓ Found expected filesystem tool: {tool_name}")
            else:
                test_summary.log(f"✗ Missing expected filesystem tool: {tool_name}", "WARNING")
        
        # Assertions
        assert len(all_proxy_tools) > 0, "No tools found in proxy"
        assert any(tool.name in expected_native_tools for tool in all_proxy_tools if hasattr(tool, 'name')), "No native proxy tools found"
        
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "PASSED", duration, f"Found {len(all_proxy_tools)} tools")
        
    except Exception as e:
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "FAILED", duration, str(e))
        test_summary.log(f"Test {test_name} failed: {e}", "ERROR")
        raise

@pytest.mark.asyncio
async def test_native_proxy_tools(proxy_client):
    """Test native proxy tools like list_managed_servers and get_server_tools."""
    test_name = "test_native_proxy_tools"
    start_time = time.time()
    
    try:
        test_summary.log(f"=== Starting {test_name} ===")
        
        # Test list_managed_servers
        test_summary.log("Testing 'list_managed_servers'...")
        managed_servers_result = await proxy_client.call_tool("list_managed_servers")
        managed_servers = parse_tool_result(managed_servers_result)
        
        # Validate managed servers result
        if isinstance(managed_servers, dict) and managed_servers.get("mcp_proxy_error"):
            raise AssertionError(f"Error from 'list_managed_servers': {managed_servers.get('detail')}")
        
        assert isinstance(managed_servers, list), f"Expected list, got {type(managed_servers)}"
        assert len(managed_servers) > 0, "No managed servers found"
        
        test_summary.log(f"Found {len(managed_servers)} managed servers")
        
        # Find a connected server for further testing
        target_server = None
        for server_info in managed_servers:
            if isinstance(server_info, dict) and server_info.get("status") == "connected" and server_info.get("config_enabled"):
                target_server = server_info
                break
        
        assert target_server is not None, "No connected and enabled server found for testing"
        target_server_name = target_server.get("name")
        test_summary.log(f"Using server '{target_server_name}' for get_server_tools test")
        
        # Test get_server_tools
        test_summary.log(f"Testing 'get_server_tools' for server: {target_server_name}...")
        server_tools_result = await proxy_client.call_tool(
                "get_server_tools",
                arguments={"server_name": target_server_name}
            )
        server_tools_data = parse_tool_result(server_tools_result)
        
        # Validate server tools result
        if isinstance(server_tools_data, dict) and server_tools_data.get("mcp_proxy_error"):
            raise AssertionError(f"Error from 'get_server_tools': {server_tools_data.get('detail')}")
        
        assert isinstance(server_tools_data, dict), f"Expected dict, got {type(server_tools_data)}"
        assert "tools" in server_tools_data, "No 'tools' key in server tools response"
        
        tools_list = server_tools_data.get("tools", [])
        assert len(tools_list) > 0, f"No tools found for server '{target_server_name}'"
        
        test_summary.log(f"Server '{target_server_name}' has {len(tools_list)} tools")
        
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "PASSED", duration, f"Tested {len(managed_servers)} servers, {len(tools_list)} tools")
        
    except Exception as e:
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "FAILED", duration, str(e))
        test_summary.log(f"Test {test_name} failed: {e}", "ERROR")
        raise

@pytest.mark.asyncio
async def test_github_tools(proxy_client):
    """Test GitHub MCP server tools via proxy."""
    test_name = "test_github_tools"
    start_time = time.time()
    
    try:
        test_summary.log(f"=== Starting {test_name} ===")
        
        # Test github_get_me (assuming it's available as GITHUB_get_me)
        github_tool_name = "GITHUB_get_me"
        
        # Check if the tool exists
        all_tools = await proxy_client.list_tools()
        tool_names = [tool.name for tool in all_tools if hasattr(tool, 'name')]
        
        if github_tool_name not in tool_names:
            duration = time.time() - start_time
            test_summary.add_test_result(test_name, "SKIPPED", duration, f"GitHub tool '{github_tool_name}' not found")
            pytest.skip(f"GitHub tool '{github_tool_name}' not found. Available tools: {tool_names}")
        
        test_summary.log(f"Testing GitHub tool: {github_tool_name}")
        
        # Call the GitHub get_me tool - try without arguments first, then with empty dict
        try:
            result = await proxy_client.call_tool(github_tool_name)
        except Exception as e1:
            test_summary.log(f"Call without arguments failed: {e1}, trying with empty arguments dict...")
            try:
                result = await proxy_client.call_tool(github_tool_name, arguments={})
            except Exception as e2:
                test_summary.log(f"Call with empty arguments failed: {e2}, trying with None arguments...")
                result = await proxy_client.call_tool(github_tool_name, arguments=None)
        
        parsed_result = parse_tool_result(result)
        
        test_summary.log(f"GitHub get_me result type: {type(parsed_result)}")
        test_summary.log(f"GitHub get_me result: {parsed_result}")
        
        # Basic validation - result should not be None and should not be an error
        assert parsed_result is not None, "GitHub get_me returned None"
        
        if isinstance(parsed_result, dict) and parsed_result.get("mcp_proxy_error"):
            raise AssertionError(f"Error from GitHub tool: {parsed_result.get('detail')}")
        
        # If it's successful, it should contain user information
        if isinstance(parsed_result, dict):
            # GitHub user info typically contains login, id, etc.
            test_summary.log("✓ GitHub get_me tool executed successfully")
        else:
            test_summary.log(f"✓ GitHub get_me tool executed, returned: {type(parsed_result)}")
        
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "PASSED", duration, "GitHub tool executed successfully")
            
    except Exception as e:
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "FAILED", duration, str(e))
        test_summary.log(f"Test {test_name} failed: {e}", "ERROR")
        raise

@pytest.mark.asyncio
async def test_filesystem_tools(proxy_client):
    """Test filesystem MCP server tools via proxy."""
    test_name = "test_filesystem_tools"
    start_time = time.time()
    
    try:
        test_summary.log(f"=== Starting {test_name} ===")
        
        # Test filesystem_list_directory
        filesystem_tool_name = "filesystem_list_directory"
        
        # Check if the tool exists
        all_tools = await proxy_client.list_tools()
        tool_names = [tool.name for tool in all_tools if hasattr(tool, 'name')]
        
        if filesystem_tool_name not in tool_names:
            duration = time.time() - start_time
            test_summary.add_test_result(test_name, "SKIPPED", duration, f"Filesystem tool '{filesystem_tool_name}' not found")
            pytest.skip(f"Filesystem tool '{filesystem_tool_name}' not found. Available tools: {tool_names}")
        
        test_summary.log(f"Testing filesystem tool: {filesystem_tool_name}")
        
        # Call the filesystem list_directory tool with different argument formats
        try:
            # Try with path as direct kwarg
            result = await proxy_client.call_tool(filesystem_tool_name, path=".")
        except Exception as e1:
            test_summary.log(f"Call with path kwarg failed: {e1}, trying with arguments dict...")
            try:
                result = await proxy_client.call_tool(filesystem_tool_name, arguments={"path": "."})
            except Exception as e2:
                test_summary.log(f"Call with arguments dict failed: {e2}, trying without path...")
                result = await proxy_client.call_tool(filesystem_tool_name)
        
        parsed_result = parse_tool_result(result)
        
        test_summary.log(f"Filesystem list_directory result type: {type(parsed_result)}")
        
        # Basic validation
        assert parsed_result is not None, "Filesystem list_directory returned None"
        
        if isinstance(parsed_result, dict) and parsed_result.get("mcp_proxy_error"):
            raise AssertionError(f"Error from filesystem tool: {parsed_result.get('detail')}")
        
        # Filesystem list_directory should return directory contents
        if isinstance(parsed_result, (list, dict)):
            test_summary.log("✓ Filesystem list_directory tool executed successfully")
            test_summary.log(f"Directory listing result: {parsed_result}")
        else:
            test_summary.log(f"✓ Filesystem list_directory tool executed, returned: {type(parsed_result)}")
        
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "PASSED", duration, "Filesystem tool executed successfully")
            
    except Exception as e:
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "FAILED", duration, str(e))
        test_summary.log(f"Test {test_name} failed: {e}", "ERROR")
        raise

@pytest.mark.asyncio
async def test_tool_call_integration(proxy_client):
    """Integration test that verifies the overall proxy functionality."""
    test_name = "test_tool_call_integration"
    start_time = time.time()
    
    try:
        test_summary.log(f"=== Starting {test_name} ===")
        
        # 1. List all tools
        all_tools = await proxy_client.list_tools()
        assert len(all_tools) > 0, "No tools available"
        
        # 2. List managed servers
        servers_result = await proxy_client.call_tool("list_managed_servers")
        servers = parse_tool_result(servers_result)
        assert isinstance(servers, list) and len(servers) > 0, "No managed servers"
        
        # 3. Get tools for each connected server
        for server_info in servers:
            if isinstance(server_info, dict) and server_info.get("status") == "connected":
                server_name = server_info.get("name")
                tools_result = await proxy_client.call_tool(
                    "get_server_tools",
                    arguments={"server_name": server_name}
                )
                tools_data = parse_tool_result(tools_result)
                
                if isinstance(tools_data, dict) and "tools" in tools_data:
                    server_tools = tools_data["tools"]
                    test_summary.log(f"Server '{server_name}' has {len(server_tools)} tools")
                    
                    # Verify that each server tool has a corresponding proxied tool
                    for tool_info in server_tools:
                        if isinstance(tool_info, dict) and "name" in tool_info:
                            original_tool_name = tool_info["name"]
                            proxied_tool_name = f"{server_name}_{original_tool_name}"
                            
                            # Check if the proxied tool exists in the main tools list
                            tool_names = [tool.name for tool in all_tools if hasattr(tool, 'name')]
                            if proxied_tool_name in tool_names:
                                test_summary.log(f"✓ Verified proxied tool exists: {proxied_tool_name}")
                            else:
                                test_summary.log(f"✗ Proxied tool missing: {proxied_tool_name}", "WARNING")
        
        test_summary.log("✓ Integration test completed")
        
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "PASSED", duration, "Integration test completed successfully")
        
    except Exception as e:
        duration = time.time() - start_time
        test_summary.add_test_result(test_name, "FAILED", duration, str(e))
        test_summary.log(f"Test {test_name} failed: {e}", "ERROR")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])

# Global test session finalizer
@pytest.fixture(scope="session", autouse=True)
def finalize_test_summary():
    """Automatically write test summary at the end of the test session."""
    yield  # Run all tests first
    
    # Write the summary after all tests are done
    try:
        test_summary.write_summary()
    except Exception as e:
        print(f"Failed to write test summary: {e}") 