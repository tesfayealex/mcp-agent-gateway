from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, BaseTool, FunctionTool
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
# Remove the old MCP imports that don't work with the proxy
# from llama_index.tools.mcp import McpToolSpec, BasicMCPClient

# Add fastmcp imports like in test_proxy_server.py
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        system_prompt: Optional[str] = "You are a helpful development assistant with access to GitHub tools and a local knowledge base. When users ask about repositories, commits, files, or other GitHub operations, use the appropriate GitHub tools (LOCAL_GITHUB_MCP_*). When they ask for general development information, use the LocalKnowledgeBaseSearch tool. Always be specific about what repository/owner information you need for GitHub operations. Maintain context from previous messages in the conversation.",
        custom_query_embedder: Optional[GeminiCustomEmbedding] = None,
        mcp_client: Optional[Client] = None,
    ):
        self.mcp_proxy_url = mcp_proxy_url
        self.rag_query_engine = rag_query_engine
        self.custom_llm = custom_llm
        self.custom_query_embedder = custom_query_embedder
        self._mcp_client = mcp_client
        self._conversation_history = []

        logger.info(f"Initializing DevAssistantAgent with LLM: {custom_llm.model_name}")
        logger.info(f"MCP Proxy URL: {mcp_proxy_url}")

        Settings.llm = self.custom_llm
        
        self.agent = FunctionAgent(
            tools=tools,
            llm=self.custom_llm,
            system_prompt=system_prompt,
            verbose=True
        )
        logger.info(f"DevAssistantAgent initialized with {len(tools)} tools.")
        
        # Log available tools for debugging
        logger.info("Available tools:")
        for tool in tools:
            logger.info(f"  - {tool.metadata.name}: {tool.metadata.description}")

    @classmethod
    async def create(
        cls,
        mcp_proxy_url: str,
        rag_query_engine: BaseQueryEngine,
        custom_llm: GeminiCustomLLM,
        system_prompt: Optional[str] = "You are a helpful development assistant with access to GitHub tools and a local knowledge base. When users ask about repositories, commits, files, or other GitHub operations, use the appropriate GitHub tools (LOCAL_GITHUB_MCP_*). When they ask for general development information, use the LocalKnowledgeBaseSearch tool. Always be specific about what repository/owner information you need for GitHub operations. Maintain context from previous messages in the conversation.",
        custom_query_embedder: Optional[GeminiCustomEmbedding] = None,
    ) -> "DevAssistantAgent":
        """
        Asynchronously creates and initializes the DevAssistantAgent, including fetching MCP tools.
        """
        # Create MCP proxy client like in test_proxy_server.py
        transport = StreamableHttpTransport(url=mcp_proxy_url)
        mcp_client = Client(transport=transport)

        try:
            # Initialize and connect the client
            logger.info("Connecting to MCP client...")
            await mcp_client.__aenter__()  # Manually enter the context manager
            logger.info("MCP client connected successfully")

            # <<<< TEMP TEST CALL >>>>
            try:
                logger.info("[CREATE_AGENT_TEST] Attempting test call to list_managed_servers within create()...")
                test_servers_result = await mcp_client.call_tool("list_managed_servers")
                logger.info(f"[CREATE_AGENT_TEST] Test call list_managed_servers result: {test_servers_result}")
            except Exception as e_test_call:
                logger.error(f"[CREATE_AGENT_TEST] Test call to list_managed_servers FAILED: {e_test_call}", exc_info=True)
                # Optionally, re-raise or handle if this is critical for agent creation
            # <<<< END TEMP TEST CALL >>>>
            
            tools = await cls._setup_tools_async(mcp_proxy_url, rag_query_engine)
            
            agent_instance = cls(
                mcp_proxy_url=mcp_proxy_url,
                rag_query_engine=rag_query_engine,
                custom_llm=custom_llm,
                tools=tools,
                system_prompt=system_prompt,
                custom_query_embedder=custom_query_embedder,
                mcp_client=mcp_client
            )
            
            return agent_instance
        except Exception as e:
            logger.error(f"Failed to create DevAssistantAgent: {e}", exc_info=True)
            # Clean up client on failure
            if mcp_client:
                try:
                    await mcp_client.__aexit__(None, None, None)
                except:
                    pass
            raise # Re-raise the exception to indicate failure

    @staticmethod # Make it static as it's called from create before instance exists
    async def _setup_tools_async(
        mcp_proxy_url: str, 
        rag_query_engine: BaseQueryEngine,
        # mcp_client: Client # No longer pass the main client here for tool execution
    ) -> List[BaseTool]:
        """Asynchronously sets up the tools for the agent using the MCP proxy."""
        all_tools: List[BaseTool] = []

        # 1. MCP Proxy Tools (using fastmcp client like test_proxy_server.py)
        # We need a temporary client here just to discover servers and tools.
        # Tool execution will create its own client.
        temp_transport = StreamableHttpTransport(url=mcp_proxy_url)
        temp_mcp_client = Client(transport=temp_transport)

        try:
            async with temp_mcp_client: # Ensure temp client is properly managed
                logger.info(f"Using temporary MCP client to discover managed servers at {mcp_proxy_url}...")
                managed_servers_result = await temp_mcp_client.call_tool("list_managed_servers")
                
                managed_servers = []
                if isinstance(managed_servers_result, list) and len(managed_servers_result) > 0:
                    # Handle TextContent response like in test file
                    from mcp.types import TextContent
                    if isinstance(managed_servers_result[0], TextContent):
                        try:
                            json_text = managed_servers_result[0].text
                            parsed_servers = json.loads(json_text)
                            if isinstance(parsed_servers, list):
                                managed_servers = parsed_servers
                                logger.info(f"Found {len(managed_servers)} managed servers")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse servers JSON: {e}")
                    elif all(isinstance(item, dict) for item in managed_servers_result):
                        managed_servers = managed_servers_result

                # For each connected server, create proxy tools
                for server_info in managed_servers:
                    if not isinstance(server_info, dict):
                        continue
                        
                    server_name = server_info.get("name")
                    if server_info.get("status") == "connected" and server_info.get("config_enabled"):
                        logger.info(f"Creating proxy tools for server: {server_name}")
                        
                        # Get server tools
                        server_tools_result = await temp_mcp_client.call_tool(
                            "get_server_tools",
                            arguments={"server_name": server_name}
                        )
                        
                        server_tools_data = None
                        if isinstance(server_tools_result, list) and len(server_tools_result) > 0:
                            from mcp.types import TextContent
                            if isinstance(server_tools_result[0], TextContent):
                                try:
                                    json_text = server_tools_result[0].text
                                    server_tools_data = json.loads(json_text)
                                except json.JSONDecodeError:
                                    pass
                        elif isinstance(server_tools_result, dict):
                            server_tools_data = server_tools_result

                        if server_tools_data and "tools" in server_tools_data:
                            tools_list = server_tools_data.get("tools", [])
                            logger.info(f"Server {server_name} has {len(tools_list)} tools")
                            
                            # Create FunctionTool for each server tool
                            for tool_info in tools_list:
                                tool_name = tool_info.get("name")
                                tool_description = tool_info.get("description", f"Tool {tool_name} from {server_name}")
                                tool_input_schema = tool_info.get("inputSchema", {})
                                
                                logger.info(f"Creating tool {tool_name} with schema: {tool_input_schema}")
                                
                                # Create a closure to capture server_name and tool_name
                                # It now also captures mcp_proxy_url to create a client per call
                                def create_proxy_tool_fn(proxy_url_for_call, srv_name, tl_name, schema):
                                    properties = schema.get("properties", {})
                                    
                                    if not properties:
                                        def proxy_tool_fn() -> str:
                                            logger.info(f"🔧 EXECUTING TOOL (no-params): {tl_name} on server {srv_name}")
                                            
                                            async def async_call_isolated():
                                                logger.info(f"🔧 [ASYNC_CALL_ISO ENTRY] For {tl_name} in thread {threading.get_ident()}. Proxy URL: {proxy_url_for_call}")
                                                # Create new transport and client inside the async function in the new thread
                                                transport = StreamableHttpTransport(url=proxy_url_for_call)
                                                isolated_mcp_client = Client(transport=transport)
                                                async with isolated_mcp_client: # Manage client lifecycle here
                                                    logger.info(f"🔧 [ASYNC_CALL_ISO PRE-AWAIT] Isolated MCP Client: {isolated_mcp_client}")
                                                    try:
                                                        result = await isolated_mcp_client.call_tool(
                                                            "call_server_tool",
                                                            arguments={
                                                                "server_name": srv_name,
                                                                "tool_name": tl_name,
                                                                "arguments": {}
                                                            }
                                                        )
                                                    except Exception as e_call:
                                                        logger.error(f"🔧 [ASYNC_CALL_ISO AWAIT ERROR] MCP call for {tl_name} failed: {e_call}", exc_info=True)
                                                        raise
                                                    logger.info(f"🔧 [ASYNC_CALL_ISO POST-AWAIT] MCP call completed for {tl_name}. Result: {type(result)}")
                                                    return result

                                            def run_async_in_thread_isolated():
                                                logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO ENTRY] For {tl_name}. Thread: {threading.get_ident()}")
                                                new_loop = None
                                                try:
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO PRE-NEW-LOOP] For {tl_name}")
                                                    new_loop = asyncio.new_event_loop()
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO POST-NEW-LOOP] For {tl_name}. Loop: {new_loop}")
                                                    asyncio.set_event_loop(new_loop)
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO POST-SET-LOOP] For {tl_name}")
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO PRE-RUN-UNTIL-COMPLETE] For {tl_name}")
                                                    result = new_loop.run_until_complete(async_call_isolated())
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO POST-RUN-UNTIL-COMPLETE] For {tl_name}. Result type: {type(result)}")
                                                    return result
                                                except Exception as e_thread_run:
                                                    logger.error(f"🔧 [RUN_ASYNC_IN_THREAD_ISO ERROR] For {tl_name}: {e_thread_run}", exc_info=True)
                                                    raise # Re-raise to be caught by the main try/except
                                                finally:
                                                    if new_loop:
                                                        logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO PRE-CLOSE-LOOP] For {tl_name}")
                                                        new_loop.close()
                                                        logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO POST-CLOSE-LOOP] For {tl_name}")
                                            
                                            try:
                                                logger.info(f"🔧 [PROXY_TOOL_FN PRE-EXECUTOR] For {tl_name}. Preparing to submit run_async_in_thread_isolated.")
                                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                                    logger.info(f"🔧 [PROXY_TOOL_FN IN-EXECUTOR] For {tl_name}. Submitting to executor.")
                                                    future = executor.submit(run_async_in_thread_isolated)
                                                    logger.info(f"🔧 [PROXY_TOOL_FN POST-SUBMIT] For {tl_name}. Future: {future}. Waiting for result...")
                                                    result = future.result(timeout=60) # Increased timeout slightly
                                                    logger.info(f"🔧 [PROXY_TOOL_FN POST-RESULT] For {tl_name}. Result type: {type(result)}")
                                            except concurrent.futures.TimeoutError:
                                                logger.error(f"🔧 Timeout (60s) waiting for isolated {tl_name} to complete")
                                                return f"Error: Tool {tl_name} timed out"
                                            except Exception as e:
                                                logger.error(f"🔧 Error in isolated async thread for {tl_name}: {e}", exc_info=True)
                                                return f"Error executing {tl_name} in thread: {e}"
                                            # Process the result (same as before)
                                            logger.info(f"🔧 TOOL {tl_name} RAW RESULT (isolated): {result}")
                                            if isinstance(result, list) and len(result) > 0:
                                                from mcp.types import TextContent
                                                if isinstance(result[0], TextContent):
                                                    try:
                                                        result_data = json.loads(result[0].text)
                                                        if result_data.get("success"): return str(result_data.get("result", ""))
                                                        else: return f"Error: {result_data.get('error_message', 'Unknown error')}"
                                                    except json.JSONDecodeError: return str(result[0].text)
                                                else: return str(result[0])
                                            elif isinstance(result, dict):
                                                if result.get("success"): return str(result.get("result", ""))
                                                else: return f"Error: {result.get('error_message', 'Unknown error')}"
                                            return str(result)
                                        return proxy_tool_fn
                                    else: # Tool with parameters
                                        def proxy_tool_fn(**kwargs) -> str:
                                            logger.info(f"🔧 EXECUTING TOOL (with-params): {tl_name} on server {srv_name} with {kwargs}")
                                            actual_arguments = kwargs

                                            async def async_call_isolated_params():
                                                logger.info(f"🔧 [ASYNC_CALL_ISO_PARAMS ENTRY] For {tl_name} in thread {threading.get_ident()}. Proxy URL: {proxy_url_for_call}")
                                                transport = StreamableHttpTransport(url=proxy_url_for_call)
                                                isolated_mcp_client = Client(transport=transport)
                                                async with isolated_mcp_client:
                                                    logger.info(f"🔧 [ASYNC_CALL_ISO_PARAMS PRE-AWAIT] Isolated MCP Client: {isolated_mcp_client}")
                                                    try:
                                                        result = await isolated_mcp_client.call_tool(
                                                            "call_server_tool",
                                                            arguments={
                                                                "server_name": srv_name,
                                                                "tool_name": tl_name,
                                                                "arguments": actual_arguments
                                                            }
                                                        )
                                                    except Exception as e_call:
                                                        logger.error(f"🔧 [ASYNC_CALL_ISO_PARAMS AWAIT ERROR] MCP call for {tl_name} failed: {e_call}", exc_info=True)
                                                        raise
                                                    logger.info(f"🔧 [ASYNC_CALL_ISO_PARAMS POST-AWAIT] MCP call completed for {tl_name}. Result: {type(result)}")
                                                    return result

                                            def run_async_in_thread_isolated_params():
                                                logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS ENTRY] For {tl_name}. Thread: {threading.get_ident()}")
                                                new_loop = None
                                                try:
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS PRE-NEW-LOOP] For {tl_name}")
                                                    new_loop = asyncio.new_event_loop()
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS POST-NEW-LOOP] For {tl_name}. Loop: {new_loop}")
                                                    asyncio.set_event_loop(new_loop)
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS POST-SET-LOOP] For {tl_name}")
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS PRE-RUN-UNTIL-COMPLETE] For {tl_name}")
                                                    result = new_loop.run_until_complete(async_call_isolated_params())
                                                    logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS POST-RUN-UNTIL-COMPLETE] For {tl_name}. Result type: {type(result)}")
                                                    return result
                                                except Exception as e_thread_run:
                                                    logger.error(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS ERROR] For {tl_name}: {e_thread_run}", exc_info=True)
                                                    raise # Re-raise to be caught by the main try/except
                                                finally:
                                                    if new_loop:
                                                        logger.info(f"🔧 [RUN_ASYNC_IN_THREAD_ISO_PARAMS PRE-CLOSE-LOOP] For {tl_name}")
                                                        new_loop.close()
                                                        logger.info(f"🔧 [RUN_ASYNC_IN_TFsHREAD_ISO_PARAMS POST-CLOSE-LOOP] For {tl_name}")

                                            try:
                                                logger.info(f"🔧 [PROXY_TOOL_FN_PARAMS PRE-EXECUTOR] For {tl_name}. Preparing to submit run_async_in_thread_isolated_params.")
                                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                                    logger.info(f"🔧 [PROXY_TOOL_FN_PARAMS IN-EXECUTOR] For {tl_name}. Submitting to executor.")
                                                    future = executor.submit(run_async_in_thread_isolated_params)
                                                    logger.info(f"🔧 [PROXY_TOOL_FN_PARAMS POST-SUBMIT] For {tl_name}. Future: {future}. Waiting for result...")
                                                    result = future.result(timeout=60) # Increased timeout
                                                    logger.info(f"🔧 [PROXY_TOOL_FN_PARAMS POST-RESULT] For {tl_name}. Result type: {type(result)}")
                                            except concurrent.futures.TimeoutError:
                                                logger.error(f"🔧 Timeout (60s) waiting for isolated {tl_name} (params) to complete")
                                                return f"Error: Tool {tl_name} (params) timed out"
                                            except Exception as e:
                                                logger.error(f"🔧 Error in isolated async thread for {tl_name} (params): {e}", exc_info=True)
                                                return f"Error executing {tl_name} (params) in thread: {e}"
                                            
                                            # Process the result (same as before)
                                            logger.info(f"🔧 TOOL {tl_name} RAW RESULT (isolated, params): {result}")
                                            if isinstance(result, list) and len(result) > 0:
                                                from mcp.types import TextContent
                                                if isinstance(result[0], TextContent):
                                                    try:
                                                        result_data = json.loads(result[0].text)
                                                        if result_data.get("success"): return str(result_data.get("result", ""))
                                                        else: return f"Error: {result_data.get('error_message', 'Unknown error')}"
                                                    except json.JSONDecodeError: return str(result[0].text)
                                                else: return str(result[0])
                                            elif isinstance(result, dict):
                                                if result.get("success"): return str(result.get("result", ""))
                                                else: return f"Error: {result.get('error_message', 'Unknown error')}"
                                            return str(result)
                                        return proxy_tool_fn
                                
                                # Sanitize server name for valid function names
                                sanitized_server_name = server_name.replace(" ", "_").replace("-", "_")
                                
                                # Create the function tool
                                proxy_tool = FunctionTool.from_defaults(
                                    fn=create_proxy_tool_fn(mcp_proxy_url, server_name, tool_name, tool_input_schema), # Pass mcp_proxy_url
                                    name=f"{sanitized_server_name}_{tool_name}",
                                    description=f"[{server_name}] {tool_description}"
                                )
                                all_tools.append(proxy_tool)
                                logger.info(f"Added proxy tool: {sanitized_server_name}_{tool_name}")

        except Exception as e:
            logger.error(f"Failed to discover MCP proxy tools: {e}", exc_info=True)
        
        # 2. RAG Tool (remains the same)
        try:
            rag_tool = QueryEngineTool.from_defaults(
                query_engine=rag_query_engine,
                name="LocalKnowledgeBaseSearch",
                description="Search and query the local knowledge base for development-related information, documentation, and code examples. Use this when you need to find specific information from the knowledge base."
            )
            all_tools.append(rag_tool)
            logger.info("Added RAG tool (LocalKnowledgeBaseSearch).")
        except Exception as e:
            logger.error(f"Failed to create RAG tool: {e}", exc_info=True)

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
            
            agent_response = await self.agent.run(conversation_context)
            
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
            response_text = "I encountered an error trying to process your request. Please try again."
            
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