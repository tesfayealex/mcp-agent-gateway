# mcp-agent-gateway

This project aims to develop an autonomous agent system capable of interacting with various external services through the Method Call Protocol (MCP). The system leverages `fastmcp` for implementing MCP servers and agents, and is designed to be containerized using Docker for ease of setup and deployment (though Docker setup is not detailed here yet).

## Project Overview

The core idea is to build intelligent agents that can perform tasks by communicating with specialized MCP servers. These servers act as adapters, translating generic MCP calls into specific API calls for services like GitHub, Google Drive, and the local filesystem. The project involves setting up the necessary environment, understanding the communication protocols, implementing agents and servers, and integrating them into a functional system.

The central entry point for running the system is the **MCP Proxy Server** (`mcp_proxy_server/proxy_server.py`). This server manages connections to various downstream MCP servers (configured in `config.json`) and provides a unified interface for agents to discover and call their tools.

## Getting Started

This section provides instructions to set up and run the `mcp-agent-gateway` system.

### Prerequisites

-   **Python**: Version 3.9 or higher.
-   **pip**: Python package installer.
-   **Git**: For cloning the repository.
-   **(Optional) Docker**: If any of your downstream MCP servers (defined in `config.json`) are intended to run as Docker containers managed via `stdio` by the MCP Manager component.

### Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url> # Replace with your repository URL
    cd mcp-agent-gateway
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    From the `mcp-agent-gateway` root directory, install all required packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `fastapi`, `uvicorn`, `fastmcp`, `pydantic`, `python-dotenv`, `llama-index`, `google-generativeai`, and other necessary libraries for all components.

4.  **Configure Environment Variables**:
    -   Copy the sample environment file `.env.sample` to a new file named `.env` in the project root:
        ```bash
        cp .env.sample .env
        ```
    -   **Edit the `.env` file** and fill in the required values, especially:
        -   `GOOGLE_API_KEY`: Your Google API key for Gemini models (used by `DevAssistantAgent`).
        -   `MCP_PROXY_URL`: The intended URL for the MCP Proxy Server itself (e.g., `http://localhost:8100/proxy`).
        -   Any environment variables required by your downstream MCP servers as defined in `config.json` (e.g., `MCP_MANAGER_GITHUB_PAT_ENV`).
    Refer to the comments in `.env.sample` for more details on each variable.

5.  **Configure Downstream MCP Servers (`config.json`)**:
    -   Review and customize the `config.json` file in the project root.
    -   This file tells the MCP Manager (which is used by the Proxy Server) which downstream MCP servers to connect to, how to run them (if `stdio`), or where to find them (if `url`).
    -   Ensure the server names, connection types, commands, URLs, and any referenced environment variables (which you set up in `.env`) are correct for your setup.
    -   For detailed information on the structure of `config.json`, see the `mcp_manager/README.md`.

### Running the System (MCP Proxy Server)

The primary way to run the system is by starting the MCP Proxy Server. It will, in turn, initialize and manage connections to the downstream MCP servers defined in `config.json`.

1.  Ensure your virtual environment is activated and you are in the `mcp-agent-gateway` root directory.
2.  Run the MCP Proxy Server using Uvicorn:
    ```bash
    uvicorn mcp_proxy_server.proxy_server:proxy_mcp --host 0.0.0.0 --port 8100 --reload
    ```
    -   `--host 0.0.0.0`: Makes the server accessible on your network.
    -   `--port 8100`: Specifies the port. Change if needed, and ensure `MCP_PROXY_URL` in your `.env` file reflects this.
    -   `--reload`: Enables auto-reloading for development. Remove for production.

Once started, the MCP Proxy Server will be available (by default) at `http://localhost:8100`. Agents, like the `DevAssistantAgent`, can then be configured to point to its MCP endpoint (e.g., `http://localhost:8100/proxy`).

### How to Run Tests

This project uses `pytest` for running automated tests.

1.  Ensure all development dependencies are installed (they should be if you installed `requirements.txt` which ideally includes `pytest`). If not, install `pytest`:
    ```bash
    pip install pytest
    ```
2.  Navigate to the `mcp-agent-gateway` root directory.
3.  Run pytest:
    ```bash
    pytest
    ```
    Pytest will automatically discover and run tests in the `tests/` directory.

## Code Structure

The project is organized into several key directories:

-   `dev_assistant_agent_py/`: Contains the [Dev Assistant Agent](./dev_assistant_agent_py/README.md). This agent uses the MCP Proxy Server to interact with tools.
-   `mcp_manager/`: Contains the [MCP Manager](./mcp_manager/README.md) components, responsible for handling connections to individual downstream MCP servers. It is used internally by the MCP Proxy Server.
-   `mcp_proxy_server/`: Contains the [MCP Proxy Server](./mcp_proxy_server/README.md), the main entry point for the system.
-   `mock_knowledge_base/`: Contains a mock knowledge base for RAG, used by the `DevAssistantAgent`.
-   `tests/`: Contains automated tests for the system.
-   `config.json`: Configuration file for defining downstream MCP servers.
-   `.env.sample` / `.env`: Environment variable configuration.
-   `requirements.txt`: Python package dependencies.

Each component directory (`dev_assistant_agent_py`, `mcp_manager`, `mcp_proxy_server`) has its own `README.md` with detailed explanations of its internal code structure and overview.

## Key Components

    - Autonomous Agents: Intelligent entities designed to perform tasks by interacting with MCP servers (e.g., `DevAssistantAgent`).

    - MCP Servers: Services that expose external APIs (GitHub, Filesystem, Google Drive, Atlassian) via the Method Call Protocol. These are the "downstream" servers managed via `config.json`.

    - MCP Proxy Server: The central server that agents connect to. It routes requests to the appropriate downstream MCP servers.

    - MCP Manager: Component used by the Proxy Server to manage connections and discovery of downstream MCP servers.

    - Method Call Protocol (MCP): The standardized protocol used by agents to invoke methods on MCP servers. `fastmcp` is the library used for this.

    - Agent-to-Agent (A2A) Communication: (To be explored/implemented) Protocol for agents to communicate and collaborate with each other.

    - Knowledge Base (KB): A local mock knowledge base for Retrieval Augmented Generation (RAG) or other agent data needs (e.g., `mock_knowledge_base/`).

    - `fastmcp`: A framework/library used for building efficient MCP agents and servers.

    - Docker: Used for containerizing the various components (agents, servers, potentially the KB) to ensure consistent environments. (Detailed Docker setup TBD).

## Implementation Details
The project will utilize `fastmcp` for implementing both the agent logic and the MCP server functionalities. The MCP Proxy Server acts as the main interface for agents. Docker can be used to package these components, allowing the entire system to be built and run in isolated containers.

## Project Phases (Based on Tasks)
The project development is structured into several phases:

    1. Environment Setup & Protocol Study: Setting up the development environment (Python, Git, Docker), creating the mock knowledge base structure, and gaining a deep understanding of the MCP and A2A protocols and the target MCP servers. This includes ensuring all setup configurations are clearly documented in this main README and component READMEs.

    2. Basic Agent & Server Implementation: Implementing a simple agent and one or more basic MCP servers using `fastmcp`. Code should be well-explained in component READMEs.

    3. Protocol Implementation & Testing: Ensuring correct implementation of MCP `invoke_method` and response handling. Testing communication between the agent and servers via the proxy. Documenting usage examples.

    4. Integration with External Services: Connecting MCP servers to actual external service APIs (GitHub, Filesystem, etc.).

    5. Agent Capabilities Expansion: Developing agents with more complex reasoning and task execution capabilities, potentially incorporating RAG using the mock KB.

    6. Dockerization: Containerizing the agent(s), MCP server(s), and potentially the mock KB for streamlined deployment.

    7. System Testing & Refinement: Testing the integrated, containerized system and refining components based on results.

## Mock Knowledge Base
A directory structure for a mock knowledge base (`mock_knowledge_base/`) will be created to support agent capabilities requiring external information retrieval (RAG). Details on populating and accessing the mock KB will be provided separately by the `DevAssistantAgent` component.

