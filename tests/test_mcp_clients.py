import asyncio
import logging
import os
import argparse

# Adjust import path to load from the parent directory's mcp_manager package
import sys

# Calculate the path to the project root (mcp-agent-gateway)
# This assumes test_mcp_clients.py is in mcp-agent-gateway/tests/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root) # Add project root to Python path

from mcp_manager.config_loader import load_configs, ServerConfig
from mcp_manager.server_handler import MCPClientWrapper

# Basic logging configuration for the test script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__) # Get logger for this module

# Mute other loggers for cleaner test output, e.g., from fastmcp if too verbose
# logging.getLogger("fastmcp").setLevel(logging.WARNING)

async def test_single_server(server_config: ServerConfig):
    """Tests connection and a configured tool for a single MCP server."""
    logger.info(f"--- Testing Server: {server_config.name} ---")
    if not server_config.enabled:
        logger.info(f"Server '{server_config.name}' is disabled. Skipping test.")
        print(f"RESULT: {server_config.name}: SKIPPED (disabled)")
        return True # Count as passed for disabled servers in terms of test execution

    wrapper = MCPClientWrapper(server_config)
    
    logger.info(f"Attempting to connect to {server_config.name} ({server_config.connection_type})...")
    connected = await wrapper.connect()

    if not connected or wrapper.status != "connected":
        logger.error(f"Failed to connect to server '{server_config.name}'. Status: {wrapper.status}")
        print(f"RESULT: {server_config.name}: FAILED (Connection Error)")
        await wrapper.disconnect() # Ensure cleanup if partially connected
        return False

    logger.info(f"Successfully connected to {server_config.name}.")

    # Test 1: List tools
    logger.info(f"Attempting to list tools for {server_config.name}...")
    tools = await wrapper.list_tools()
    if tools is not None: # list_tools returns [] on error, or list of tools
        logger.info(f"Available tools for {server_config.name}: {tools if tools else 'No tools listed or list_tools returned empty'}")
        # We consider successful empty list_tools call as a pass for this step, actual tools depend on server.
    else: # Should not happen if list_tools is implemented to return [] on error
        logger.error(f"Failed to list tools for '{server_config.name}' (returned None). Status: {wrapper.status}")
        # wrapper.status might have changed to 'error' if list_tools failed internally
        print(f"RESULT: {server_config.name}: FAILED (List Tools Error - returned None)")
        await wrapper.disconnect()
        return False
    
    if wrapper.status != "connected": # If list_tools caused an error and changed status
        logger.error(f"Server '{server_config.name}' status changed to '{wrapper.status}' after list_tools call.")
        print(f"RESULT: {server_config.name}: FAILED (List Tools - Status Error)")
        await wrapper.disconnect()
        return False

    # Test 2: Call configured test tool
    if server_config.test_tool:
        tool_name = server_config.test_tool.name
        tool_params = server_config.test_tool.params
        logger.info(f"Attempting to call test tool '{tool_name}' with params {tool_params} for {server_config.name}...")
        
        tool_result = await wrapper.call_tool(tool_name, tool_params)
        
        if tool_result is not None:
            logger.info(f"Test tool '{tool_name}' for {server_config.name} executed. Result: {str(tool_result)[:200]}... (truncated if long)")
            # Success of the tool call is just that it executed without error and returned something.
            # Specific validation of the tool_result content is beyond this basic test.
        else:
            logger.error(f"Failed to call test tool '{tool_name}' for {server_config.name} (returned None). Status: {wrapper.status}")
            print(f"RESULT: {server_config.name}: FAILED (Test Tool Call Error)")
            await wrapper.disconnect()
            return False
        
        if wrapper.status != "connected": # If call_tool caused an error and changed status
            logger.error(f"Server '{server_config.name}' status changed to '{wrapper.status}' after call_tool '{tool_name}'.")
            print(f"RESULT: {server_config.name}: FAILED (Test Tool Call - Status Error)")
            await wrapper.disconnect()
            return False
    else:
        logger.warning(f"No test_tool configured for server '{server_config.name}'. Skipping test tool call.")

    logger.info(f"All tests passed for server '{server_config.name}'.")
    print(f"RESULT: {server_config.name}: PASSED")
    await wrapper.disconnect()
    return True

async def run_tests(args):
    logger.info("Starting MCP Client Tester...")
    
    if args.config:
        config_file = args.config
    else:
        # Default to config.json in the project root (one level up from tests/)
        config_file = os.path.join(project_root, "config.json")

    logger.info(f"Loading server configurations from: {config_file}")
    server_configs = load_configs(config_path=config_file)

    if not server_configs:
        logger.error("No server configurations found. Exiting tester.")
        return

    results = {}
    total_servers = 0
    passed_servers = 0

    for conf in server_configs:
        if args.server_name and conf.name != args.server_name:
            logger.debug(f"Skipping server '{conf.name}' as it does not match requested server '{args.server_name}'.")
            continue
        
        total_servers +=1
        test_passed = await test_single_server(conf)
        results[conf.name] = "PASSED" if test_passed else "FAILED"
        if test_passed and conf.enabled: # Only count enabled and passed as truly passed
             passed_servers +=1
        elif not conf.enabled and test_passed: # If disabled and test_single_server returned True (for skipped)
             # Don't count towards passed_servers, but also not a failure for the run.
             # The total_servers count will include it if not filtered by name.
             pass 

    logger.info("--- Test Summary ---")
    for name, result in results.items():
        logger.info(f"Server '{name}': {result}")
    
    # Adjust total_servers if a specific server_name was requested
    if args.server_name:
        if args.server_name in results:
            logger.info(f"Tested 1 server ('{args.server_name}'). Result: {results[args.server_name]}")
            if results[args.server_name] == "PASSED":
                 final_passed = 1
                 final_total = 1
            else:
                 final_passed = 0
                 final_total = 1
        else:
            logger.error(f"Requested server name '{args.server_name}' not found in configuration.")
            final_passed = 0
            final_total = 0 # Or 1, depending on how you want to report this.
    else:
        final_passed = passed_servers
        # Recalculate total_servers to be only *enabled* servers for the summary percentage
        final_total = sum(1 for sc in server_configs if sc.enabled)

    if final_total > 0:
        pass_percentage = (final_passed / final_total) * 100
        logger.info(f"Overall: {final_passed}/{final_total} enabled servers passed ({pass_percentage:.2f}%).")
    elif total_servers > 0 and final_total == 0: # All servers were disabled
        logger.info("Overall: No enabled servers to test.")
    else: # No servers configured or found by name
        logger.info("Overall: No servers were tested.")

    if final_passed < final_total:
        logger.error("Some tests failed.")
        # sys.exit(1) # Indicate failure for CI/CD pipelines if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test utility for MCP server connections and basic tool calls.")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to the JSON configuration file. Defaults to 'config.json' in the project root."
    )
    parser.add_argument(
        "-s", "--server-name",
        type=str,
        help="Optional: Test only the server with this specific name from the config file."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging for the test script and mcp_manager components."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        # Specifically set mcp_manager loggers to DEBUG if they exist
        logging.getLogger("mcp_manager").setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")
    else:
        # Keep mcp_manager logs at INFO or WARNING if not verbose testing
        logging.getLogger("mcp_manager").setLevel(logging.INFO) 

    asyncio.run(run_tests(args)) 