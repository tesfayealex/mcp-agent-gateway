import asyncio
import os
import logging
from dotenv import load_dotenv

# Assuming these custom modules are in the same directory or sys.path is configured
from custom_llm import GeminiCustomLLM
from custom_embedder import GeminiCustomEmbedding
from rag_setup import create_rag_query_engine
from agent import DevAssistantAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main function to initialize and run the Dev Assistant Agent."""
    logger.info("Starting Dev Assistant Agent setup...")

    # Load environment variables from .env file located at the project root
    # (one level up from dev_assistant_agent_py directory)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded .env file from: {dotenv_path}")

    # Retrieve configuration from environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    mcp_proxy_url = os.getenv("MCP_PROXY_URL")
    
    # Ensure MCP proxy URL is in the correct format for the SSE endpoint
    if mcp_proxy_url and not mcp_proxy_url.endswith('/sse'):
        if mcp_proxy_url.endswith('/'):
            mcp_proxy_url = mcp_proxy_url + 'sse'
        else:
            mcp_proxy_url = mcp_proxy_url + '/sse'
    
    # KNOWLEDGE_BASE_PATH from .env is relative to the project root (where .env is)
    knowledge_base_env_path = os.getenv("KNOWLEDGE_BASE_PATH", "./mock_knowledge_base")
    # Resolve its absolute path based on project root (parent of dev_assistant_agent_py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    knowledge_base_abs_path = os.path.abspath(os.path.join(project_root, knowledge_base_env_path))
    
    chat_model_name = os.getenv("GEMINI_CHAT_MODEL_NAME", "gemini-1.5-flash-latest")
    embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/embedding-001")
    
    # RAG storage directory (local to this script's location for simplicity)
    rag_storage_dir = os.path.join(os.path.dirname(__file__), "agent_rag_storage_prod")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY is not set in the .env file. Agent cannot start.")
        return
    if not mcp_proxy_url:
        # MCP Proxy URL is essential for the current agent.create design
        logger.error("MCP_PROXY_URL is not set in the .env file. Agent cannot start as it relies on MCP tool discovery.")
        return

    logger.info(f"Google API Key: {'Set' if google_api_key else 'Not Set'}")
    logger.info(f"MCP Proxy URL: {mcp_proxy_url}")
    logger.info(f"Knowledge Base Path (resolved): {knowledge_base_abs_path}")
    logger.info(f"Chat Model: {chat_model_name}, Embedding Model: {embedding_model_name}")
    logger.info(f"RAG Storage Directory: {rag_storage_dir}")

    # Ensure mock_knowledge_base exists or create a dummy one for the agent to run
    if not os.path.exists(knowledge_base_abs_path) or not os.listdir(knowledge_base_abs_path):
        logger.warning(f"Knowledge base at {knowledge_base_abs_path} is empty or missing.")
        logger.info("Creating a dummy knowledge base for demonstration purposes.")
        os.makedirs(knowledge_base_abs_path, exist_ok=True)
        with open(os.path.join(knowledge_base_abs_path, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder document. Replace with actual knowledge base content.")

    # 1. Initialize Custom LLM and Embedders
    logger.info("Initializing LLM and Embedding models...")
    try:
        custom_llm = GeminiCustomLLM(model_name=chat_model_name, api_key=google_api_key)
        # For indexing documents
        doc_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name, 
            api_key=google_api_key, 
            task_type="RETRIEVAL_DOCUMENT"
        )
        # For querying
        query_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name, 
            api_key=google_api_key, 
            task_type="RETRIEVAL_QUERY"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini models: {e}", exc_info=True)
        return
    logger.info("LLM and Embedding models initialized.")

    # 2. Setup RAG Query Engine
    logger.info("Setting up RAG query engine...")
    try:
        rag_engine = create_rag_query_engine(
            knowledge_base_path=knowledge_base_abs_path,
            custom_embedding_model=doc_embedder, # Use document embedder for indexing
            custom_llm=custom_llm, # For synthesis within RAG
            persist_dir=rag_storage_dir
        )
    except Exception as e:
        logger.error(f"Failed to create RAG query engine: {e}", exc_info=True)
        return
    logger.info("RAG query engine ready.")

    # 3. Initialize Agent
    logger.info("Initializing Dev Assistant Agent asynchronously...")
    dev_assistant = None # Initialize to None for finally block
    try:
        # Use the async factory method to create the agent
        dev_assistant = await DevAssistantAgent.create(
            mcp_proxy_url=mcp_proxy_url,
            rag_query_engine=rag_engine,
            custom_llm=custom_llm,
            custom_query_embedder=query_embedder
        )
        logger.info("Dev Assistant Agent is ready.")

        print("\nWelcome to the Dev Assistant Agent CLI!")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'help' to see available commands.")
        print("\nThe agent has access to GitHub tools and a local knowledge base.")
        print("You can ask about repositories, commits, files, issues, PRs, etc.")
        print("Example: 'get the commit history for repository owner/repo-name'\n")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting agent CLI. Goodbye!")
                    break
                elif user_input.lower() == "clear":
                    dev_assistant.clear_conversation_history()
                    print("Agent: Conversation history cleared. Starting fresh!")
                    continue
                elif user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  'exit' or 'quit' - End the session")
                    print("  'clear' - Clear conversation history")
                    print("  'help' - Show this help message")
                    print("\nExample queries:")
                    print("  'get the commit history for repository owner/repo-name'")
                    print("  'list files in repository owner/repo-name'")
                    print("  'search for issues in repository owner/repo-name'")
                    print("  'tell me about software development best practices' (uses knowledge base)")
                    continue
                if not user_input.strip():
                    continue

                print("Agent: (Processing your request...)")
                response = await dev_assistant.handle_message(user_input)
                print(f"Agent: {response}")

            except KeyboardInterrupt:
                print("\nExiting agent CLI due to interrupt. Goodbye!")
                break
            except Exception as e:
                logger.error(f"An error occurred in the interactive loop: {e}", exc_info=True)
                print("Agent: I encountered an issue. Please try again.")

    except Exception as e:
        logger.error(f"Failed to initialize or run DevAssistantAgent: {e}", exc_info=True)
    finally:
        if dev_assistant:
            await dev_assistant.close() # Ensure agent's resources are cleaned up
        logger.info("Dev Assistant Agent session ended.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error during agent startup or runtime: {e}", exc_info=True) 