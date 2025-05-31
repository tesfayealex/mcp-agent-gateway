# Dev Assistant Agent (`dev_assistant_agent_py`)

This directory contains the source code and related files for the Dev Assistant Agent, a sophisticated agent designed to assist with development tasks by leveraging external tools and a local knowledge base.

## Overview

The Dev Assistant Agent is built using Python with the LlamaIndex framework. It interacts with various development tools (e.g., GitHub) through the Method Call Protocol (MCP) by connecting to the central **MCP Proxy Server**. Key features include:

-   **MCP Tool Integration**: Dynamically discovers and utilizes tools exposed by MCP servers. It achieves this by querying the MCP Proxy Server (using its `list_managed_servers` and `get_server_tools` tools) and then calling specific tools on downstream servers via the proxy's `call_server_tool`.
-   **`fastmcp` Client**: Uses the `fastmcp.client.Client` to communicate with the MCP Proxy Server.
-   **Retrieval Augmented Generation (RAG)**: Incorporates a local RAG pipeline to answer queries using a knowledge base (configured via `KNOWLEDGE_BASE_PATH` environment variable, typically pointing to `mcp-agent-gateway/mock_knowledge_base/`). This allows the agent to provide information from project-specific documents or other relevant local data.
-   **Custom LLM and Embedding Support**: Utilizes Google's Gemini models (e.g., `gemini-1.5-flash-latest` for chat, `models/embedding-001` for embeddings) via custom LlamaIndex wrappers (`GeminiCustomLLM`, `GeminiCustomEmbedding`). These are configured using the `GOOGLE_API_KEY` and model name environment variables.
-   **Conversation History**: Maintains a history of interactions to provide contextually aware responses.
-   **Interactive CLI**: Provides a command-line interface for users to interact with the agent, started by running `run.py`.

The agent is designed to be configurable primarily through environment variables (see the main project `README.md` and `.env.sample`).

For setup, installation, and how to run this agent (by first running the main MCP Proxy Server and then this agent), please refer to the main `README.md` in the `mcp-agent-gateway` project root.

## Code Explanations

Key files and their roles within the `dev_assistant_agent_py` directory:

-   **`agent.py`**:
    -   Defines the `DevAssistantAgent` class, the core of the agent.
    -   Manages integration with LlamaIndex, including the `FunctionAgent`.
    -   Handles dynamic tool setup by connecting to the MCP Proxy Server (specified by `MCP_PROXY_URL` env var) using `fastmcp.client.Client`. It calls proxy tools like `list_managed_servers` and `get_server_tools` to discover what downstream tools are available, and then constructs LlamaIndex `FunctionTool` instances that will make calls to the proxy's `call_server_tool`.
    -   Manages conversation history for contextual understanding.
    -   Implements the `handle_message` method to process user queries and generate responses using the LLM and configured tools.
    -   Includes an asynchronous `create` factory method for proper initialization of async resources like the `fastmcp` client connection to the proxy.

-   **`run.py`**:
    -   The main executable script to start *this specific* Dev Assistant Agent (after the MCP Proxy Server is already running).
    -   Loads environment variables (e.g., `GOOGLE_API_KEY`, `MCP_PROXY_URL`, `KNOWLEDGE_BASE_PATH`).
    -   Initializes the `GeminiCustomLLM` and `GeminiCustomEmbedding` components.
    -   Sets up the RAG query engine using `create_rag_query_engine` from `rag_setup.py`.
    -   Instantiates the `DevAssistantAgent` (which connects to the running MCP Proxy Server) and provides an interactive CLI for user interaction.

-   **`custom_llm.py`**:
    -   Contains the `GeminiCustomLLM` class, a wrapper around Google's Gemini API to make it compatible with LlamaIndex's LLM interface.

-   **`custom_embedder.py`**:
    -   Contains the `GeminiCustomEmbedding` class, a wrapper for Google's Gemini Embedding API, compatible with LlamaIndex's embedding interface. Supports different task types (e.g., document retrieval, query retrieval).

-   **`rag_setup.py`**:
    -   Provides the `create_rag_query_engine` function responsible for initializing the LlamaIndex RAG pipeline.
    -   This includes loading documents from the `KNOWLEDGE_BASE_PATH`, creating an index (and persisting it to `agent_rag_storage_prod/`), and returning a query engine for the agent to use.

-   **`agent_rag_storage_prod/`**:
    -   This directory is automatically created by the RAG system (via `rag_setup.py`) to store the persisted vector index and other RAG-related data. This allows the agent to load the index quickly on subsequent runs without re-processing all documents from the knowledge base.

-   **`realtime_rag_notes.md`**:
    -   Likely a placeholder or notes file related to RAG development or features. Its specific current use should be reviewed. 