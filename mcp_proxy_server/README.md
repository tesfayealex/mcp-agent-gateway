# MCP Proxy Server (`mcp_proxy_server`)

This directory contains the MCP (Method Call Protocol) Proxy Server, a FastAPI-based application that acts as the **primary entry point** and an intermediary between MCP clients (like agents) and various downstream MCP servers.

## Overview

The MCP Proxy Server simplifies interaction with a distributed set of MCP servers by providing a single, unified endpoint. Its key responsibilities include:

-   **Unified Access Point**: Exposes a single MCP endpoint (typically configured via `MCP_PROXY_URL` environment variable, e.g., `http://localhost:8100/proxy`) for clients to discover and invoke tools on any of the connected downstream MCP servers.
-   **Integration with MCP Manager**: Internally initializes and uses the `MCPConnectionManager` (from the `mcp_manager` component) to manage all aspects of downstream server connections. This includes loading their configurations from the main `config.json` file.
-   **Tool Discovery**: Provides tools for clients to:
    -   `list_managed_servers`: Get a list of all downstream servers known to the MCP Manager and their current status and configuration enabled status.
    -   `get_server_tools`: Retrieve the list of tools and their schemas (name, description, parameters) for a specific, connected downstream server.
-   **Tool Invocation Proxying**: Provides a `call_server_tool` method that takes a `server_name`, `tool_name`, and `arguments`, and routes the call to the appropriate downstream MCP server via the embedded `MCPConnectionManager`.
-   **FastAPI and FastMCP**: Built using FastAPI for the web framework and `FastMCP` for handling MCP communication, including request/response serialization and tool registration for its own proxy-specific tools.
-   **Lifespan Management**: Uses FastAPI's lifespan events (`app_lifespan` function) to initialize the `MCPConnectionManager` on startup (which then connects to downstream servers defined in `config.json`) and gracefully shut it down (disconnecting from servers) when the proxy server stops.

This proxy is essential for agents or other MCP clients that need to interact with multiple, potentially diverse MCP servers without needing to manage individual connections or discovery for each one. The `DevAssistantAgent`, for example, connects only to this proxy server.

For setup, installation, and how to run this proxy server (which is the main way to run the entire system), please refer to the main `README.md` in the `mcp-agent-gateway` project root.

## Code Explanations

Key files and their roles within the `mcp_proxy_server` directory:

-   **`proxy_server.py`**:
    -   The main file and entry point for the MCP Proxy Server application.
    -   Initializes a `FastMCP` application instance (`proxy_mcp`). The MCP methods of this instance are exposed under the path defined by `streamable_http_path` (e.g., `/proxy`).
    -   Defines the `app_lifespan` asynchronous context manager. This is automatically managed by FastAPI/Uvicorn:
        -   On application startup: It loads the main `config.json` (from the project root), initializes the `MCPConnectionManager` (from `mcp_manager` module) with these configurations. The `MCPConnectionManager` then attempts to connect to all enabled downstream servers and starts monitoring them.
        -   On application shutdown: It calls methods on the `MCPConnectionManager` to stop monitoring and disconnect all downstream servers.
    -   Defines the proxy's own MCP tools which are exposed to clients:
        -   `list_managed_servers()`: Returns a list of downstream servers and their status by querying the embedded `MCPConnectionManager`.
        -   `get_server_tools(server_name: str)`: Fetches tool schemas for a specified downstream server, again by using the `MCPConnectionManager`.
        -   `call_server_tool(server_name: str, tool_name: str, arguments: Dict[str, Any])`: The core proxying method. It uses the `MCPConnectionManager` to find the correct `ServerHandler` for the `server_name` and then delegates the tool call to that handler, which communicates with the actual downstream MCP server.

-   **`models.py`**:
    -   Defines Pydantic models for the data structures used in the proxy server's API, particularly for the request and response bodies of its own MCP tools (`list_managed_servers`, `get_server_tools`, `call_server_tool`).
    -   Examples include `ManagedServerInfo` (describing a downstream server's state), `DownstreamToolSchema` (describing a tool on a downstream server), `ToolParameterSchema`, `ListDownstreamToolsResponse`, `ToolCallResult` (for the outcome of `call_server_tool`), and `ErrorResponse`.
    -   These models ensure clear and validated data contracts for clients (like agents) interacting with the proxy server's tools.

-   **`__init__.py`**:
    -   Makes the `mcp_proxy_server` directory a Python package, allowing Uvicorn to properly locate the application instance (e.g., `mcp_proxy_server.proxy_server:proxy_mcp`).

The MCP Proxy Server, by embedding and managing the `MCPConnectionManager`, effectively becomes the master controller for all MCP operations within the `mcp-agent-gateway` system. 