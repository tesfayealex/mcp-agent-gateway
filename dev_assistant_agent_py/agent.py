from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, BaseTool, FunctionTool
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
# Remove the old MCP imports that don't work with the proxy
# from llama_index.tools.mcp import McpToolSpec, BasicMCPClient

# Add fastmcp imports like in test_proxy_server.py
from fastmcp.client import Client
from fastmcp.client.transports import SSETransport

# from .custom_llm import GeminiCustomLLM # Package import
from custom_llm import GeminiCustomLLM # Direct import for dev/testing
# from .custom_embedder import GeminiCustomEmbedding # Package import
from custom_embedder import GeminiCustomEmbedding # Direct import

from typing import List, Optional, Any, Dict
import logging
import os
import json
from dotenv import load_dotenv
import asyncio # For async handle_message
import aiohttp # Import aiohttp
import threading
import concurrent.futures
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def retry_with_exponential_backoff(
    operation, 
    max_retries: int = 2, 
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_errors: tuple = (Exception,)
):
    """
    Retry an async operation with exponential backoff.
    
    Args:
        operation: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        retryable_errors: Tuple of exception types that should trigger a retry
    
    Returns:
        Result of the operation if successful
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return await operation()
        except retryable_errors as e:
            last_exception = e
            
            # Don't retry certain types of errors
            error_str = str(e).lower()
            if any(no_retry in error_str for no_retry in ["authentication", "unauthorized", "forbidden", "not found"]):
                logger.info(f"Non-retryable error encountered: {e}")
                raise e
            
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                break
                
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    
    raise last_exception

class DevAssistantAgent:
    """
    An agent that uses MCP (Model Context Protocol) tools via proxy and a local RAG query engine.
    Maintains conversation history for better context awareness.
    """
    agent: FunctionAgent
    mcp_proxy_url: str
    rag_query_engine: BaseQueryEngine
    custom_llm: GeminiCustomLLM
    # Optional: If you need a different embedder for queries vs. indexing
    custom_query_embedder: Optional[GeminiCustomEmbedding] = None
    _mcp_client: Optional[Client] = None # Store the MCP proxy client
    _conversation_history: List[ChatMessage] = [] # Store conversation history

    # Make __init__ synchronous and simple, moving async setup to a factory
    def __init__(
        self,
        mcp_proxy_url: str,
        rag_query_engine: BaseQueryEngine,
        custom_llm: GeminiCustomLLM,
        tools: List[BaseTool], # Tools will now be passed in
        system_prompt: Optional[str] = "You are a helpful development assistant with access to MCP tools and a local knowledge base. Use MCP tools for GitHub operations, filesystem access, and other development tasks. Use the LocalKnowledgeBaseSearch tool for general development information. Always be specific about what information you need for operations. Maintain context from previous messages in the conversation.",
        custom_query_embedder: Optional[GeminiCustomEmbedding] = None,
        mcp_client: Optional[Client] = None,
    ):
        self.mcp_proxy_url = mcp_proxy_url
        self.rag_query_engine = rag_query_engine
        self.custom_llm = custom_llm
        self.custom_query_embedder = custom_query_embedder
        self._mcp_client = mcp_client
        self._conversation_history = []
        
        # Create the agent with the provided tools and system prompt
        self.agent = FunctionAgent(
            tools=tools,
            llm=custom_llm,
            system_prompt="You are a helpful assistant with access to various tools including local knowledge search and external APIs. When you need information to complete a task, use the available tools to gather that information first, then proceed with the main task. You can chain multiple tool calls together in a single response. For example, if you need a username for a GitHub operation, get it first using GITHUB_get_me, then use that information for subsequent operations. Always provide all required parameters for tool calls.",
            verbose=True
        )
        logger.info(f"DevAssistantAgent initialized with {len(tools)} tools")

    @classmethod
    async def create(
        cls,
        mcp_proxy_url: str,
        rag_query_engine: BaseQueryEngine,
        custom_llm: GeminiCustomLLM,
        system_prompt: Optional[str] = "You are a helpful development assistant with access to MCP tools and a local knowledge base. Use MCP tools for GitHub operations, filesystem access, and other development tasks. Use the LocalKnowledgeBaseSearch tool for general development information. Always be specific about what information you need for operations. Maintain context from previous messages in the conversation.",
        custom_query_embedder: Optional[GeminiCustomEmbedding] = None,
    ) -> "DevAssistantAgent":
        """
        Create a DevAssistantAgent instance with tools discovered from the MCP proxy.
        Uses SSE transport and standard MCP calls.
        """
        logger.info(f"Creating DevAssistantAgent with MCP proxy at: {mcp_proxy_url}")
        
        # Setup tools asynchronously using SSE transport
        tools = await cls._setup_tools_async(mcp_proxy_url, rag_query_engine)
        
        # Create MCP proxy client using SSE transport
        transport = SSETransport(url=mcp_proxy_url)
        mcp_client = Client(transport=transport)
        
        return cls(
            mcp_proxy_url=mcp_proxy_url,
            rag_query_engine=rag_query_engine,
            custom_llm=custom_llm,
            tools=tools,
            system_prompt=system_prompt,
            custom_query_embedder=custom_query_embedder,
            mcp_client=mcp_client
        )

    @staticmethod # Make it static as it's called from create before instance exists
    async def _setup_tools_async(
        mcp_proxy_url: str, 
        rag_query_engine: BaseQueryEngine,
    ) -> List[BaseTool]:
        """
        Discover and create all available tools from the MCP proxy and RAG engine.
        Uses standard MCP calls without server_name references.
        """
        all_tools = []
        
        # 1. Add the local RAG query engine tool first
        local_knowledge_tool = QueryEngineTool.from_defaults(
            query_engine=rag_query_engine,
            name="LocalKnowledgeBaseSearch",
            description="Search the local knowledge base for development-related information, documentation, and best practices. Use this when users ask general development questions."
        )
        all_tools.append(local_knowledge_tool)
        
        # 2. MCP proxy tools - simplified approach using SSE transport
        try:
            logger.info(f"🔧 Discovering MCP proxy tools from {mcp_proxy_url}")
            
            # Create temporary client to discover tools
            temp_transport = SSETransport(url=mcp_proxy_url)
            temp_mcp_client = Client(transport=temp_transport)
            
            async with temp_mcp_client:
                # Get all available tools from the proxy
                proxy_tools = await temp_mcp_client.list_tools()
                logger.info(f"🔧 Found {len(proxy_tools)} tools from proxy")
                
                for tool_info in proxy_tools:
                    if hasattr(tool_info, 'name') and hasattr(tool_info, 'description'):
                        tool_name = tool_info.name
                        tool_description = tool_info.description
                        
                        logger.info(f"🔧 Creating proxy tool: {tool_name}")
                        
                        # Get tool schema to understand expected parameters
                        tool_schema = getattr(tool_info, 'inputSchema', None) or getattr(tool_info, 'parameters', {})
                        logger.info(f"🔧 Tool {tool_name} schema: {tool_schema}")
                        
                        # Create proxy tool function that calls the tool directly
                        def create_proxy_tool_fn(tl_name, proxy_url, expected_schema):
                            def proxy_tool_fn(**kwargs) -> str:
                                logger.info(f"🔧 EXECUTING TOOL: {tl_name} with args: {kwargs}")
                                
                                # Use kwargs directly as the arguments
                                # Remove any 'kwargs' wrapper if it exists from the LLM
                                if 'kwargs' in kwargs and len(kwargs) == 1:
                                    # LLM might wrap all args in a 'kwargs' field, unwrap it
                                    inner_kwargs = kwargs['kwargs']
                                    if isinstance(inner_kwargs, dict):
                                        parsed_args = inner_kwargs
                                        logger.info(f"🔧 Unwrapped kwargs field: {parsed_args}")
                                    elif isinstance(inner_kwargs, str):
                                        try:
                                            parsed_args = json.loads(inner_kwargs)
                                            logger.info(f"🔧 Parsed JSON from kwargs field: {parsed_args}")
                                        except json.JSONDecodeError as e:
                                            logger.error(f"🔧 Failed to parse JSON from kwargs field: {inner_kwargs}, error: {e}")
                                            
                                            # Enhanced parsing logic for various string formats
                                            parsed_args = {}
                                            
                                            # Try to parse as query string format like "query=user:tesfayealex"
                                            if '=' in inner_kwargs:
                                                try:
                                                    # Simple parsing for key=value pairs
                                                    pairs = inner_kwargs.split('&') if '&' in inner_kwargs else [inner_kwargs]
                                                    for pair in pairs:
                                                        if '=' in pair:
                                                            key, value = pair.split('=', 1)
                                                            # Decode URL-encoded values if needed
                                                            key = key.strip()
                                                            value = value.strip()
                                                            # Handle quoted values
                                                            if value.startswith('"') and value.endswith('"'):
                                                                value = value[1:-1]
                                                            parsed_args[key] = value
                                                    logger.info(f"🔧 Parsed query string format: {parsed_args}")
                                                except Exception as parse_e:
                                                    logger.error(f"🔧 Failed to parse as query string: {parse_e}")
                                            
                                            # If we still don't have args, try to extract from the tool name or description
                                            if not parsed_args:
                                                # For search tools, assume the string is a query
                                                if 'search' in tl_name.lower():
                                                    parsed_args = {'q': inner_kwargs}
                                                    logger.info(f"🔧 Assumed search query: {parsed_args}")
                                                elif 'query' in tl_name.lower():
                                                    parsed_args = {'query': inner_kwargs}
                                                    logger.info(f"🔧 Assumed query parameter: {parsed_args}")
                                                else:
                                                    # Last resort: try common parameter names
                                                    logger.warning(f"🔧 Using fallback parameter parsing for: {inner_kwargs}")
                                                    # Check if it looks like just a value without a key
                                                    if inner_kwargs and not any(char in inner_kwargs for char in ['=', '{', '[', ':']):
                                                        # Try common parameter names based on tool name
                                                        if 'github' in tl_name.lower():
                                                            if 'search' in tl_name.lower() and 'repo' in tl_name.lower():
                                                                parsed_args = {'query': inner_kwargs}
                                                            elif 'user' in tl_name.lower():
                                                                parsed_args = {'q': f"user:{inner_kwargs}"}
                                                            else:
                                                                parsed_args = {'query': inner_kwargs}
                                                        else:
                                                            parsed_args = {'query': inner_kwargs}
                                                        logger.info(f"🔧 Applied fallback parsing: {parsed_args}")
                                    else:
                                        # Handle non-string, non-dict values
                                        logger.warning(f"🔧 Unexpected kwargs type: {type(inner_kwargs)}, value: {inner_kwargs}")
                                        parsed_args = {}
                                else:
                                    # Use arguments directly
                                    parsed_args = kwargs
                                    logger.info(f"🔧 Using direct args: {parsed_args}")
                                
                                # Validate that we have some arguments for tools that require them
                                if not parsed_args:
                                    logger.warning(f"🔧 No arguments parsed for tool {tl_name}, this may cause validation errors")
                                    # For search tools, provide a helpful error message
                                    if 'search' in tl_name.lower():
                                        return f"Error: Search tools require a query parameter. Please provide what you want to search for."
                                
                                async def async_call():
                                    try:
                                        transport = SSETransport(url=proxy_url)
                                        async with Client(transport=transport) as client:
                                            try:
                                                # Always use call_tool with arguments parameter
                                                if parsed_args:
                                                    result = await client.call_tool(tl_name, arguments=parsed_args)
                                                else:
                                                    result = await client.call_tool(tl_name)
                                                return result
                                            except Exception as e:
                                                logger.error(f"🔧 Error calling tool {tl_name}: {e}")
                                                return {"error": f"Error executing tool {tl_name}: {str(e)}"}
                                    except Exception as e:
                                        logger.error(f"🔧 Error connecting to MCP proxy for tool {tl_name}: {e}")
                                        return {"error": f"Error connecting to MCP proxy: {str(e)}"}

                                def run_in_thread():
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    try:
                                        result = new_loop.run_until_complete(async_call())
                                        return result
                                    finally:
                                        new_loop.close()
                                
                                try:
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(run_in_thread)
                                        result = future.result(timeout=60)
                                except concurrent.futures.TimeoutError:
                                    logger.error(f"🔧 Timeout (60s) waiting for {tl_name} to complete")
                                    return f"Error: Tool {tl_name} timed out after 60 seconds"
                                except Exception as e:
                                    logger.error(f"🔧 Error in thread execution for {tl_name}: {e}")
                                    return f"Error executing {tl_name}: {str(e)}"
                                
                                logger.info(f"🔧 TOOL {tl_name} RESULT: {result}")
                                
                                # Process result
                                if isinstance(result, dict):
                                    if result.get("error"):
                                        return f"Error: {result['error']}"
                                    elif result.get("result"):
                                        return str(result["result"])
                                    else:
                                        return str(result)
                                else:
                                    return str(result)
                            
                            return proxy_tool_fn
                        
                        # Create the tool with proper parameters handling
                        proxy_tool = FunctionTool.from_defaults(
                            fn=create_proxy_tool_fn(tool_name, mcp_proxy_url, tool_schema),
                            name=tool_name,
                            description=tool_description,
                        )
                        
                        all_tools.append(proxy_tool)
                        
        except Exception as e:
            logger.error(f"🚨 Error setting up MCP proxy tools: {e}")
            # Continue with just local tools if proxy fails
        
        logger.info(f"🔧 Total tools created: {len(all_tools)}")
        return all_tools

    async def close(self):
        """Clean up resources, like the MCP client session."""
        if self._mcp_client:
            logger.info("Closing MCP client for DevAssistantAgent.")
            try:
                # Properly close the MCP client connection
                await self._mcp_client.__aexit__(None, None, None)
                logger.info("MCP client closed successfully.")
            except Exception as e:
                logger.warning(f"Error during MCP client cleanup: {e}")
            finally:
                self._mcp_client = None

    async def handle_message(self, user_query: str) -> str:
        """
        Handles a user query by invoking the agent and returning the response.
        Maintains conversation history for context awareness.

        Args:
            user_query: The query from the user.

        Returns:
            The agent's textual response.
        """
        logger.info(f"Handling user query: {user_query}")
        logger.info(f"Current conversation history has {len(self._conversation_history)} messages")

        # Add user message to conversation history
        user_message = ChatMessage(role=MessageRole.USER, content=user_query)
        self._conversation_history.append(user_message)

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
            # Log the conversation context being sent to the agent
            logger.info("Sending conversation context to agent:")
            for i, msg in enumerate(self._conversation_history):
                logger.info(f"  {i+1}. {msg.role.value}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            
            # Use the full conversation history with the agent
            # Create a single string with the conversation context
            conversation_context = "\n".join([
                f"{msg.role.value}: {msg.content}" for msg in self._conversation_history
            ])
            
            logger.info("Invoking agent with conversation context...")
            logger.info(f"Full conversation context being sent: {conversation_context}")
            
            # Use retry mechanism for agent execution
            async def run_agent():
                return await self.agent.run(conversation_context)
            
            agent_response = await retry_with_exponential_backoff(
                run_agent,
                max_retries=2,
                base_delay=1.0,
                retryable_errors=(Exception,)  # Most errors are retryable except for specific ones handled in retry function
            )
            
            logger.info(f"Agent returned response type: {type(agent_response)}")
            logger.info(f"Agent response: {agent_response}")
            
            response_text = str(agent_response)
            logger.info(f"Agent response converted to string: {response_text}")
            
            # Add assistant response to conversation history
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
            self._conversation_history.append(assistant_message)
            
            # Trim conversation history if it gets too long (keep last 20 messages)
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]
                logger.info("Trimmed conversation history to last 20 messages")
            
        except Exception as e:
            logger.error(f"Error during agent query execution: {e}", exc_info=True)
            
            # Provide more specific error messages based on the error type
            if "AttributeError" in str(e) and "block_reason" in str(e):
                response_text = "I encountered a temporary issue with the language model. Let me try a different approach to help you."
            elif "validation error" in str(e).lower() and "missing_argument" in str(e).lower():
                response_text = "I had trouble understanding the required parameters for that operation. Could you please rephrase your request or provide more specific details?"
            elif "tool" in str(e).lower() and ("timeout" in str(e).lower() or "error" in str(e).lower()):
                response_text = "I encountered an issue with one of the tools I was trying to use. This might be a temporary problem. Please try your request again, or let me know if you'd like to try a different approach."
            elif "connection" in str(e).lower() or "proxy" in str(e).lower():
                response_text = "I'm having trouble connecting to some of my tools right now. I can still help with general questions and information from my knowledge base. What would you like to know?"
            else:
                response_text = "I encountered an unexpected error while processing your request. Please try rephrasing your question or let me know if you need help with something specific."
            
            # Add helpful suggestions based on common use cases
            if any(keyword in user_query.lower() for keyword in ["github", "repository", "repo", "pull request", "pr", "issue"]):
                response_text += "\n\nIf you're looking for GitHub-related help, I can assist with:\n- Searching repositories\n- Creating issues or pull requests\n- Reviewing code\n- Managing branches\n\nPlease try rephrasing your request with more specific details."
            elif any(keyword in user_query.lower() for keyword in ["file", "directory", "folder", "read", "write", "edit"]):
                response_text += "\n\nIf you're looking for file operations, I can help with:\n- Reading file contents\n- Creating or editing files\n- Listing directory contents\n- Searching for files\n\nPlease try rephrasing your request with more specific file paths or operations."
            elif any(keyword in user_query.lower() for keyword in ["code", "programming", "development", "debug"]):
                response_text += "\n\nI can help with programming and development questions using my knowledge base. Please feel free to ask about specific programming concepts, best practices, or debugging approaches."
            
            # Still add the error response to history for context
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
            self._conversation_history.append(assistant_message)
        finally:
            # Restore the original embed_model if it was changed
            if self.custom_query_embedder:
                Settings.embed_model = original_embed_model
                logger.info("Restored original embedder to Settings.")

        logger.info(f"Final response to user: {response_text}")
        return response_text
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self._conversation_history = []
        logger.info("Conversation history cleared")

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

    async def main_async(): # Make the main execution async
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found for agent test. Please set it in .env at project root.")
            return

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

        from rag_setup import create_rag_query_engine
        
        rag_engine = create_rag_query_engine(
            knowledge_base_path=kb_full_path,
            custom_embedding_model=doc_embedder,
            custom_llm=dev_llm,
            persist_dir=rag_storage_dir
        )
        logger.info("RAG engine initialized for agent.")

        dev_assistant = None
        try:
            logger.info("Creating DevAssistantAgent asynchronously...")
            dev_assistant = await DevAssistantAgent.create(
                mcp_proxy_url=mcp_proxy_url_env,
                rag_query_engine=rag_engine,
                custom_llm=dev_llm,
                custom_query_embedder=query_embedder
            )
            logger.info("DevAssistantAgent created successfully.")

            test_query = "Tell me about Gemini. What is its primary function?"
            logger.info(f"\nSending query to agent: '{test_query}'")
            response = await dev_assistant.handle_message(test_query)
            logger.info(f"\nAgent's final response:\n{response}")

        except Exception as e:
            logger.error(f"Error in agent execution: {e}", exc_info=True)
        finally:
            if dev_assistant:
                await dev_assistant.close() # Ensure session is closed
            logger.info("Agent test completed or errored.")

    if not google_api_key: # Check before starting async main
        logger.error("GOOGLE_API_KEY not found for agent test. Please set it in .env at project root.")
    else:
        asyncio.run(main_async()) 