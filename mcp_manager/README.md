# MCP Manager (`mcp_manager`)

This directory contains the MCP (Method Call Protocol) Manager, a component responsible for discovering, connecting to, and monitoring various downstream MCP servers. It is primarily used internally by the MCP Proxy Server.

## Overview

The MCP Manager acts as the core logic for handling connections to multiple MCP servers. Its primary functions, which are leveraged by the MCP Proxy Server, include:

-   **Configuration Loading**: Reads server definitions from a JSON configuration file (typically `config.json` located in the project root, passed to it by the Proxy Server).
-   **Connection Management**: Establishes and maintains connections to the configured MCP servers. It supports two types of connections:
    -   `stdio`: For MCP servers that communicate over standard input/output (e.g., local scripts or Docker containers). The manager handles launching these subprocesses.
    -   `url`: For MCP servers accessible via HTTP/HTTPS endpoints.
-   **Server Monitoring**: Periodically checks the health and status of connected MCP servers by calling a designated `test_tool` for each server.
-   **Tool Discovery and Caching**: Fetches the list of available tools from each connected MCP server and caches this information.
-   **Graceful Shutdown**: Provides methods to cleanly disconnect from servers and stop monitoring, used by the Proxy Server during its shutdown sequence.

It uses the `fastmcp` library for low-level MCP communication (via `fastmcp.client.Client`) and Pydantic for configuration validation (`ServerConfig` models loaded from `config.json`).

For setup, installation, and how to run the overall system (which involves running the MCP Proxy Server that uses this manager), please refer to the main `README.md` in the `mcp-agent-gateway` project root.

## Code Explanations

Key files and their roles within the `mcp_manager` directory:

-   **`main.py`**:
    -   Provides an entry point to run the MCP Manager *stand-alone*, which might be useful for testing or specific deployment scenarios. However, in the primary architecture, the `MCPConnectionManager` from this module is instantiated and used directly by the `MCPProxyServer`.
    -   If run directly, it parses command-line arguments (like config path and monitoring interval), loads server configurations, and starts the `MCPConnectionManager`.

-   **`config_loader.py`**:
    -   Defines Pydantic models (`ServerConfig`, `StdioConfig`, `UrlConfig`, `AuthConfig`, `TestToolConfig`) for validating the structure of the server entries in `config.json`.
    -   The `load_configs` function reads the JSON file (path provided by the caller, e.g., MCP Proxy Server or `main.py`), validates each server entry against the Pydantic models, and returns a list of `ServerConfig` objects.
    -   It uses `python-dotenv` to enable the *potential* for loading environment variables that can be referenced within the configurations (e.g., for API tokens for `stdio_config.env_vars_to_pass` or `authentication.token_env_var`). The actual resolution of these environment variable values into the connection parameters is typically handled by the `ServerHandler` at the time of connection.

-   **`connection_manager.py`**:
    -   Contains the `MCPConnectionManager` class, the central orchestrator for this component.
    -   Manages a collection of `ServerHandler` instances, one for each configured and enabled MCP server from `config.json`.
    -   Orchestrates connecting to all enabled servers (`connect_all_servers`) by delegating to individual `ServerHandler` instances.
    -   Manages the server monitoring loop (`start_monitoring`), which periodically triggers health checks in each `ServerHandler`.
    -   Handles graceful disconnection (`disconnect_all_servers`, `stop_monitoring`).
    -   Provides methods for the MCP Proxy Server to get status, tool lists, and invoke tools on downstream servers (e.g., `get_server_handler`, `get_server_status_info`, `get_all_server_tools`, `call_tool_on_server`).

-   **`server_handler.py`**:
    -   Contains the `ServerHandler` class.
    -   Each instance is responsible for managing the lifecycle and communication with a single downstream MCP server (as defined by one entry in `config.json`).
    -   Handles establishing the connection using `fastmcp.client.Client` based on the `connection_type` (`stdio` or `url`).
        -   For `stdio`, it launches the server subprocess using `asyncio.create_subprocess_exec`, passing resolved environment variables from `stdio_config.env_vars_to_pass`.
        -   For `url`, it prepares the `StreamableHttpTransport`, including any authentication token resolved from `authentication.token_env_var`.
    -   Performs health checks by calling the server's configured `test_tool`.
    -   Manages reconnection logic (retries, delays) if a connection drops or fails.
    -   Fetches and caches the list of tools (`list_tools()`) available from the connected server.
    -   Provides the `call_tool()` method to execute a specific tool on its managed server.

-   **`__init__.py`**:
    -   Makes the `mcp_manager` directory a Python package. 