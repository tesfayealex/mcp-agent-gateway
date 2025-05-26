from llama_index.core import Settings
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import QueryEngineTool, BaseTool
from llama_index.core.query_engine import BaseQueryEngine
from llama_index_tools_mcp import McpToolSpec, BasicMCPClient

# from .custom_llm import GeminiCustomLLM # Package import
from custom_llm import GeminiCustomLLM # Direct import for dev/testing
# from .custom_embedder import GeminiCustomEmbedding # Package import
from custom_embedder import GeminiCustomEmbedding # Direct import

from typing import List, Optional, Any
import logging
import os
from dotenv import load_dotenv
import asyncio # For async handle_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DevAssistantAgent:
    """
    An agent that uses MCP (Model Context Protocol) tools and a local RAG query engine.
    """
    agent: FunctionAgent
    mcp_proxy_url: str
    rag_query_engine: BaseQueryEngine
    custom_llm: GeminiCustomLLM
    # Optional: If you need a different embedder for queries vs. indexing
    custom_query_embedder: Optional[GeminiCustomEmbedding] = None 

    def __init__(
        self,
        mcp_proxy_url: str,
        rag_query_engine: BaseQueryEngine,
        custom_llm: GeminiCustomLLM,
        system_prompt: Optional[str] = "You are a helpful development assistant. Use available tools to answer questions about software development, code, and related topics. Prioritize fetching real-time data with tools when appropriate, and use the local knowledge base for general context or supplementary information.",
        custom_query_embedder: Optional[GeminiCustomEmbedding] = None,
    ):
        """
        Initializes the DevAssistantAgent.

        Args:
            mcp_proxy_url: URL of the MCP Proxy server.
            rag_query_engine: Initialized RAG query engine.
            custom_llm: Custom Gemini LLM instance.
            system_prompt: System prompt for the agent.
            custom_query_embedder: Optional custom Gemini embedding model for RAG queries.
                                   If None, the one used for indexing (from RAG engine's setup)
                                   or global Settings.embed_model will be used for queries.
        """
        self.mcp_proxy_url = mcp_proxy_url
        self.rag_query_engine = rag_query_engine
        self.custom_llm = custom_llm
        self.custom_query_embedder = custom_query_embedder

        logger.info(f"Initializing DevAssistantAgent with LLM: {custom_llm.model_name}")
        logger.info(f"MCP Proxy URL: {mcp_proxy_url}")

        # Set the LLM globally for LlamaIndex components that might need it (e.g., RAG synthesis)
        # or ensure it's passed explicitly to all components.
        Settings.llm = self.custom_llm
        # The embed_model for indexing should have been set when rag_query_engine was created.
        # If a specific query embedder is provided, it will be set before querying.

        tools = self._setup_tools()
        
        self.agent = FunctionAgent.from_tools(
            tools,
            llm=self.custom_llm,
            system_prompt=system_prompt,
            verbose=True # Enable verbose logging for agent actions
        )
        logger.info(f"DevAssistantAgent initialized with {len(tools)} tools.")

    def _setup_tools(self) -> List[BaseTool]:
        """Sets up the tools for the agent, including MCP tools and RAG tool."""
        all_tools: List[BaseTool] = [] 

        # 1. MCP Tools
        try:
            logger.info(f"Connecting to MCP Proxy at {self.mcp_proxy_url} to get tools...")
            mcp_client = BasicMCPClient(self.mcp_proxy_url)
            # Note: McpToolSpec might make synchronous calls during initialization or to_tool_list.
            # If this needs to be async, the setup might need adjustment.
            mcp_tool_spec = McpToolSpec(client=mcp_client)
            discovered_mcp_tools = mcp_tool_spec.to_tool_list()
            all_tools.extend(discovered_mcp_tools)
            logger.info(f"Discovered {len(discovered_mcp_tools)} MCP tools.")
        except Exception as e:
            logger.error(f"Failed to connect to MCP Proxy or discover tools: {e}")
            # Decide if this is a fatal error or if the agent can proceed without MCP tools.
            # For now, we'll let it proceed and log the error.

        # 2. RAG Tool
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=self.rag_query_engine,
            name="LocalKnowledgeBaseSearch",
            description="Searches and retrieves information from the local knowledge base. Use this for general context, documentation, or when explicit file/issue details are not provided for other tools."
        )
        all_tools.append(rag_tool)
        logger.info("Added RAG tool (LocalKnowledgeBaseSearch).")
        
        return all_tools

    async def handle_message(self, user_query: str) -> str:
        """
        Handles a user query by invoking the agent and returning the response.

        Args:
            user_query: The query from the user.

        Returns:
            The agent's textual response.
        """
        logger.info(f"Handling user query: {user_query}")

        # If a specific query embedder is configured, set it in LlamaIndex Settings
        # This is important because the RAG tool (QueryEngineTool) will use Settings.embed_model by default.
        original_embed_model = Settings.embed_model
        if self.custom_query_embedder:
            logger.info(f"Setting query-specific embedder: {self.custom_query_embedder.model_name} with task_type: {self.custom_query_embedder.task_type}")
            Settings.embed_model = self.custom_query_embedder
        elif original_embed_model:
             logger.info(f"Using existing Settings.embed_model for query: {original_embed_model.model_name}")
        else:
            logger.warning("No specific query embedder and Settings.embed_model is not set. RAG queries might fail or use an unexpected model.")

        try:
            # Use achat for asynchronous response from the agent
            agent_response = await self.agent.achat(user_query)
            response_text = str(agent_response)
            logger.info(f"Agent raw response: {agent_response}")
        except Exception as e:
            logger.error(f"Error during agent query execution: {e}", exc_info=True)
            response_text = "I encountered an error trying to process your request. Please try again."
        finally:
            # Restore the original embed_model if it was changed
            if self.custom_query_embedder:
                Settings.embed_model = original_embed_model
                logger.info("Restored original embedder to Settings.")

        logger.info(f"Final response to user: {response_text}")
        return response_text

if __name__ == '__main__':
    # This example assumes .env is in the parent directory of dev_assistant_agent_py
    # and mock_knowledge_base is also in the parent directory.
    # It also assumes an MCP proxy is running at the specified URL.

    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    mcp_proxy_url_env = os.getenv("MCP_PROXY_URL", "http://localhost:8001/sse") # Default if not in .env
    
    knowledge_base_relative_path = os.getenv("KNOWLEDGE_BASE_PATH", "../mock_knowledge_base")
    kb_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', knowledge_base_relative_path.lstrip("./")))
    
    chat_model_name = os.getenv("GEMINI_CHAT_MODEL_NAME", "gemini-1.5-flash-latest")
    embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/embedding-001")

    # Create a dummy knowledge base if it doesn't exist for testing
    if not os.path.exists(kb_full_path):
        os.makedirs(kb_full_path, exist_ok=True)
        with open(os.path.join(kb_full_path, "agent_test_doc.txt"), "w") as f:
            f.write("The agent uses MCP tools and RAG. Gemini is an LLM.")
        logger.info(f"Created dummy knowledge base for agent test at: {kb_full_path}")

    rag_storage_dir = os.path.join(os.path.dirname(__file__), "test_agent_rag_storage")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found for agent test. Please set it in .env at project root.")
    else:
        # 1. Initialize Custom LLM and Embedders
        logger.info("Initializing models for agent...")
        dev_llm = GeminiCustomLLM(model_name=chat_model_name, api_key=google_api_key)
        doc_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name, 
            api_key=google_api_key, 
            task_type="RETRIEVAL_DOCUMENT"
        )
        query_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name, 
            api_key=google_api_key, 
            task_type="RETRIEVAL_QUERY"
        )

        # 2. Setup RAG Query Engine (using the RAG setup script functionality)
        # For testing, we can call it directly. Make sure rag_setup.py is importable or in same dir.
        from rag_setup import create_rag_query_engine # Assumes rag_setup.py is in the same directory
        
        rag_engine = create_rag_query_engine(
            knowledge_base_path=kb_full_path,
            custom_embedding_model=doc_embedder, # Use document embedder for indexing
            custom_llm=dev_llm, # LLM for synthesis in RAG
            persist_dir=rag_storage_dir
        )
        logger.info("RAG engine initialized for agent.")

        # 3. Initialize Agent
        dev_assistant = DevAssistantAgent(
            mcp_proxy_url=mcp_proxy_url_env,
            rag_query_engine=rag_engine,
            custom_llm=dev_llm,
            custom_query_embedder=query_embedder # Pass the query-specific embedder
        )
        logger.info("DevAssistantAgent initialized.")

        # 4. Test with a query
        async def run_agent_query():
            # This query assumes your MCP proxy might have a 'get_github_issue' tool
            # or it will try to use RAG.
            test_query = "Tell me about Gemini. What is its primary function?"
            # test_query = "What can you tell me from the knowledge base about agents?"
            # test_query = "Fetch details for GitHub issue #123 in repo 'test/repo'." # Needs MCP tool
            
            logger.info(f"\nSending query to agent: '{test_query}'")
            response = await dev_assistant.handle_message(test_query)
            logger.info(f"\nAgent's final response:\n{response}")

        asyncio.run(run_agent_query())
        logger.info("Agent test completed.") 