from google import genai
from google.genai import types
import llama_index

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.tools.types import BaseTool
from pydantic import Field, PrivateAttr

from typing import Any, List, Sequence, Optional, AsyncIterator, Iterator, Dict, Tuple
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)

# Map LlamaIndex roles to Gemini roles
def _to_gemini_role(role: MessageRole) -> str:
    if role == MessageRole.USER:
        return "user"
    elif role == MessageRole.ASSISTANT:
        return "model"
    elif role == MessageRole.SYSTEM: # Gemini API doesn't have a distinct system role in history
        return "user" # Treat system messages as user messages for history
    elif role == MessageRole.FUNCTION: # Gemini API uses 'function' for function calls, 'model' for function responses
        return "function" # This might need adjustment based on specific function calling implementation
    elif role == MessageRole.TOOL: # Gemini API uses 'tool_code' and 'tool_code_output'
        return "model" # Or handle specifically if using Gemini function calling
    else:
        return "user"

class GeminiCustomLLM(FunctionCallingLLM):
    # Declare as Pydantic fields
    model_name: str = Field(description="The name of the Gemini model to use.")
    api_key: str = Field(description="The API key for Google Gemini.", exclude=True) # Exclude from serialization if needed
    temperature: float = Field(default=0.7, description="The temperature for sampling.")
    top_p: float = Field(default=1.0, description="The top-p for sampling.")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the LLM.")

    # Internal state, not part of the Pydantic model schema for configuration
    _client: Any = PrivateAttr(default=None)
    _is_chat_model: bool = PrivateAttr(default=True)
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str = "",
        temperature: float = 0.7,
        top_p: float = 1.0,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any # To catch any other unexpected Pydantic fields or allow extensibility
    ):
        # Prepare data for Pydantic model initialization
        # Ensure callback_manager is properly defaulted if None
        data_for_super = {
            "model_name": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "top_p": top_p,
            "system_prompt": system_prompt,
            "callback_manager": callback_manager or CallbackManager([]),
            **kwargs # Pass through any other kwargs
        }
        
        # Call super().__init__ first to let Pydantic initialize fields
        super().__init__(**data_for_super)

        # Now self.model_name, self.api_key etc. are initialized by Pydantic
        # Proceed with genai configuration
        self._client = genai.Client(api_key=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,  # Example, adjust based on the specific Gemini model
            num_output=2048,      # Example, adjust
            is_chat_model=self._is_chat_model,
            is_function_calling_model=True,
            model_name=self.model_name,
        )

    def _prepare_gemini_chat_history(self, messages: Sequence[ChatMessage]) -> List[dict]:
        history = []
        for msg in messages:
            role = _to_gemini_role(msg.role)
            # Skip system messages if they are handled by system_instruction at model level
            if msg.role == MessageRole.SYSTEM and self.system_prompt: 
                continue 
            history.append({
                'role': role, 
                'parts': [{'text': msg.content}]
            })
        return history

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Prepare configuration with system instruction if available
        config = None
        if self.system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p
            )

        # Convert messages to the new format
        contents = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and self.system_prompt:
                continue  # System messages handled in config
            contents.append(msg.content)

        # Use the new generate_content API
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents[-1] if contents else "",  # Send the last message
            config=config
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.text,
            ),
            raw=response,
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # For async, we'll run the sync method in an executor since the new client doesn't have async methods yet
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.chat(messages, **kwargs)
        )
        return result

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        # Prepare configuration
        config = None
        if self.system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p
            )

        # Convert messages to the new format
        contents = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and self.system_prompt:
                continue
            contents.append(msg.content)

        # Use streaming generate_content
        stream = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=contents[-1] if contents else "",
            config=config
        )

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in stream:
                delta = chunk.text if hasattr(chunk, 'text') else ""
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=delta,
                    raw=chunk,
                )
        return gen()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        # For async streaming, run in executor
        loop = asyncio.get_event_loop()
        
        async def gen() -> ChatResponseGen:
            # Run the sync streaming in executor
            sync_gen = await loop.run_in_executor(
                None,
                lambda: self.stream_chat(messages, **kwargs)
            )
            for chunk in sync_gen:
                yield chunk
        
        return gen()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        # Prepare configuration
        config = None
        if self.system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p
            )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return CompletionResponse(text=response.text, raw=response)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.complete(prompt, formatted, **kwargs)
        )
        return result

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        # Prepare configuration
        config = None
        if self.system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p
            )

        stream = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in stream:
                delta = chunk.text if hasattr(chunk, 'text') else ""
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)
        return gen()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        loop = asyncio.get_event_loop()
        
        async def gen() -> CompletionResponseGen:
            sync_gen = await loop.run_in_executor(
                None,
                lambda: self.stream_complete(prompt, formatted, **kwargs)
            )
            for chunk in sync_gen:
                yield chunk
        
        return gen()

    def _llamaindex_tool_to_gemini_function_declaration(self, tool: BaseTool) -> 'genai.types.FunctionDeclaration':
        """Converts a LlamaIndex BaseTool to a Gemini FunctionDeclaration."""
        # FunctionTool.metadata.to_openai_tool() gives a good base
        # Example:
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "get_weather",
        #         "description": "Get the current weather",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
        #                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature"}
        #             },
        #             "required": ["location"]
        #         }
        #     }
        # }
        # Gemini expects:
        # FunctionDeclaration(
        #     name="get_current_weather",
        #     description="Get the current weather in a given location",
        #     parameters={
        #         "type": "object",
        #         "properties": {
        #             "location": {
        #               "type": "string",
        #               "description": "The city name..."
        #            }
        #         },
        #     },
        # )

        try:
            openai_tool_schema = tool.metadata.to_openai_tool()
            logger.debug(f"Converting tool {tool.metadata.name} with OpenAI schema: {openai_tool_schema}")
        except Exception as e:
            logger.error(f"Error converting tool {tool.metadata.name} to OpenAI schema: {e}")
            # Create a simple fallback schema
            openai_tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description or f"Tool: {tool.metadata.name}",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        
        # Basic type mapping from OpenAPI to Gemini/JSON Schema
        # This might need to be more robust for complex schemas
        def map_type(param_type: str) -> str:
            if param_type == "integer":
                return "INTEGER" # Gemini uses uppercase for some types
            if param_type == "number":
                return "NUMBER"
            if param_type == "boolean":
                return "BOOLEAN"
            if param_type == "array":
                return "ARRAY"
            if param_type == "object":
                return "OBJECT"
            return "STRING" # Default to STRING

        gemini_params = {}
        if "parameters" in openai_tool_schema["function"] and openai_tool_schema["function"]["parameters"].get("properties"):
            gemini_params = {
                "type": "OBJECT", # Gemini seems to use uppercase for OBJECT type
                "properties": {},
                "required": openai_tool_schema["function"]["parameters"].get("required", [])
            }
            for name, details in openai_tool_schema["function"]["parameters"]["properties"].items():
                prop_details = {"type": map_type(details.get("type", "string"))}
                if "description" in details:
                    prop_details["description"] = details["description"]
                if "enum" in details:
                    # Gemini enums are list of strings, even for numbers/integers as per docs
                    prop_details["enum"] = [str(e) for e in details["enum"]]
                if "items" in details and details.get("type") == "array": # Handle array items type
                    item_type = details["items"].get("type", "string")
                    prop_details["items"] = {"type": map_type(item_type)}

                gemini_params["properties"][name] = prop_details
        
        function_declaration = genai.types.FunctionDeclaration(
            name=openai_tool_schema["function"]["name"],
            description=openai_tool_schema["function"]["description"],
            parameters=gemini_params if gemini_params.get("properties") else None # Pass None if no properties
        )
        
        logger.debug(f"Created Gemini function declaration for {tool.metadata.name}: {function_declaration}")
        return function_declaration

    def chat_with_tools(
        self, tools: List[BaseTool], user_msg: Optional[str] = None, chat_history: Optional[List[ChatMessage]] = None, verbose: bool = False, allow_parallel_tool_calls: bool = False, **kwargs: Any
    ) -> ChatResponse:
        logger.info(f"chat_with_tools called with {len(tools)} tools.")
        logger.info(f"Available tools: {[tool.metadata.name for tool in tools]}")
        
        if not tools:
            # Fallback to regular chat if no tools are provided
            logger.info("No tools provided, falling back to regular chat")
            if user_msg:
                messages = [ChatMessage(role=MessageRole.USER, content=user_msg)]
                if chat_history:
                    messages = chat_history + messages
                return self.chat(messages, **kwargs)
            elif chat_history:
                return self.chat(chat_history, **kwargs)
            else:
                return ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="No input provided.")
                )

        gemini_tools_declarations = [
            self._llamaindex_tool_to_gemini_function_declaration(tool) for tool in tools
        ]
        gemini_sdk_tool = genai.types.Tool(function_declarations=gemini_tools_declarations)
        
        logger.info(f"Prepared {len(gemini_tools_declarations)} tool declarations for Gemini")
        
        # Prepare messages from chat_history and user_msg
        messages = []
        if chat_history:
            messages.extend(chat_history)
        if user_msg:
            if isinstance(user_msg, str):
                messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
            else:
                messages.append(user_msg)
        
        logger.info(f"Sending {len(messages)} messages to Gemini with tools")
        
        # Convert LlamaIndex ChatMessage to Gemini Content format
        gemini_contents = []
        for msg in messages:
            role = "user" # Default for Gemini
            if msg.role == MessageRole.MODEL or msg.role == MessageRole.ASSISTANT:
                role = "model"
            
            parts = [genai.types.Part(text=msg.content)]
            
            # Handle previous tool calls and responses if they are in additional_kwargs
            # This part is crucial for multi-turn tool use.
            # LlamaIndex agent might put tool results in a USER message, or tool requests in ASSISTANT message.
            if msg.additional_kwargs:
                if "tool_calls" in msg.additional_kwargs and msg.role == MessageRole.ASSISTANT:
                    # This was a previous assistant message requesting a tool call
                    # Convert LlamaIndex ToolSelection back to Gemini FunctionCall Part
                    # This assumes ToolSelection has id, name, and args (kwargs)
                    fc_parts = []
                    for tc in msg.additional_kwargs["tool_calls"]:
                        fc_parts.append(genai.types.Part(
                            function_call=genai.types.FunctionCall(name=tc.tool_name, args=tc.tool_kwargs)
                        ))
                    if fc_parts:
                         parts = fc_parts # Replace text part if tool_calls exist
                elif msg.role == MessageRole.TOOL: # LlamaIndex uses MessageRole.TOOL for tool results
                    # This is a tool response message
                    # Gemini expects a Part.from_function_response
                    # name should be the function name, response is a dict with "content" or "result"
                    tool_call_id = msg.additional_kwargs.get("tool_call_id", "") # Not directly used by Gemini like OpenAI id
                    tool_name = msg.additional_kwargs.get("name", "") # LlamaIndex might put tool name here
                    
                    # Gemini\'s Part.from_function_response expects `name` and `response` (a dict)
                    # The `response` dict should contain the actual content from the tool.
                    # LlamaIndex puts the raw tool output string in msg.content
                    # We need to structure it as Gemini expects, usually {"content": actual_tool_output_string }
                    # or {"result": actual_tool_output_string}
                    if tool_name and msg.content:
                         parts = [genai.types.Part.from_function_response(
                            name=tool_name,
                            response={"content": msg.content} # Gemini expects a dict here
                        )]


            gemini_contents.append(genai.types.Content(parts=parts, role=role))
        
        logger.debug(f"Gemini contents prepared: {gemini_contents}")
        logger.debug(f"Gemini tools: {gemini_sdk_tool}")

        # Prepare configuration with system instruction and tools
        config = None
        if self.system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                tools=[gemini_sdk_tool]  # Add tools to config instead of as separate parameter
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                tools=[gemini_sdk_tool]  # Add tools to config instead of as separate parameter
            )
        
        logger.info("Calling Gemini API with tools enabled...")
        try:
            gemini_response = self._client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config  # Pass tools via config instead of separate parameter
            )
            logger.info("Gemini API call successful")
            logger.debug(f"Gemini raw response: {gemini_response}")
        except Exception as e:
            logger.error(f"Error calling Gemini model: {e}")
            # Return a ChatResponse with an error message
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"Error during LLM call: {str(e)}"
                )
            )

        response_message_content = ""
        tool_selections = []

        # The Gemini response structure for function calls:
        # response.candidates[0].content.parts has a list of Part objects.
        # If it's a function call, the part will have a `function_call` attribute.
        # response.text will be None if a function call is made.
        # response.candidates[0].finish_reason == "TOOL_CODE" (or similar) might indicate tool use.
        
        # Check if Gemini API suggested a tool call
        # A single model turn can include multiple tool calls (parallel function calling)
        if gemini_response.candidates and gemini_response.candidates[0].content and gemini_response.candidates[0].content.parts:
            for part in gemini_response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    logger.info(f"🔧 TOOL CALL DETECTED: {fc.name} with args {fc.args}")
                    tool_selections.append(
                        ToolSelection(
                            tool_id=str(uuid.uuid4()), # LlamaIndex expects a tool_id
                            tool_name=fc.name,
                            tool_kwargs=dict(fc.args), # Convert from Gemini\'s ArgType to dict
                        )
                    )
                elif hasattr(part, 'text') and part.text: # Check if part has text
                    response_message_content += part.text


        if not tool_selections and not response_message_content:
             # If no tool calls and no text, it might be a block or empty response
            finish_reason = gemini_response.candidates[0].finish_reason if gemini_response.candidates else "UNKNOWN"
            prompt_feedback = gemini_response.prompt_feedback if hasattr(gemini_response, 'prompt_feedback') else None
            logger.warning(f"No tool selections or text content from Gemini. Finish reason: {finish_reason}. Prompt Feedback: {prompt_feedback}")
            # Check for content safety blocks
            if (gemini_response.prompt_feedback and 
                hasattr(gemini_response.prompt_feedback, 'block_reason') and
                gemini_response.prompt_feedback.block_reason):
                response_message_content = f"Blocked by API. Reason: {gemini_response.prompt_feedback.block_reason}"
                if hasattr(gemini_response.prompt_feedback, 'block_reason_message') and gemini_response.prompt_feedback.block_reason_message:
                    response_message_content += f" Message: {gemini_response.prompt_feedback.block_reason_message}"
            else:
                response_message_content = "Received an empty or unexpected response from the LLM."


        additional_kwargs = {}
        if tool_selections:
            additional_kwargs["tool_calls"] = tool_selections
            # As per LlamaIndex, if tool_calls are present, content should be None or empty
            response_message_content = None 
            logger.info(f"🔧 RETURNING {len(tool_selections)} TOOL CALLS: {[ts.tool_name for ts in tool_selections]}")
        else:
            logger.info("📝 NO TOOL CALLS - Returning text response")


        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_message_content,
                additional_kwargs=additional_kwargs,
            ),
            raw=gemini_response,
        )

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Extract tool calls from a ChatResponse.
        
        This method is required by FunctionCallingLLM base class.
        """
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        
        if not tool_calls and error_on_no_tool_call:
            raise ValueError("No tool calls found in response")
        
        return tool_calls

    async def achat_with_tools(
        self, tools: List[BaseTool], user_msg: Optional[str] = None, chat_history: Optional[List[ChatMessage]] = None, verbose: bool = False, allow_parallel_tool_calls: bool = False, **kwargs: Any
    ) -> ChatResponse:
        # For now, just a wrapper around the synchronous version.
        # TODO: Implement true async version using _model.generate_content_async
        logger.info("achat_with_tools called, forwarding to synchronous chat_with_tools.")
        return self.chat_with_tools(tools, user_msg, chat_history, verbose, allow_parallel_tool_calls, **kwargs)
        

    def stream_chat_with_tools(
        self, tools: List[BaseTool], user_msg: Optional[str] = None, chat_history: Optional[List[ChatMessage]] = None, verbose: bool = False, allow_parallel_tool_calls: bool = False, **kwargs: Any
    ) -> ChatResponseGen:
        # TODO: Implement true streaming version for tool calling.
        # This is complex because tool calls might not be streamable in parts,
        # or might arrive as a single chunk, while text responses are streamed.
        logger.warning("stream_chat_with_tools called, but full streaming for tool calls is not yet implemented. Falling back to non-streaming chat_with_tools and wrapping its single response.")
        
        chat_response = self.chat_with_tools(tools, user_msg, chat_history, verbose, allow_parallel_tool_calls, **kwargs)
        
        async def single_response_generator():
            yield chat_response
        
        # This is not a true generator for streaming tokens but fulfills the type
        # by yielding a single complete ChatResponse.
        # A proper implementation would use self._model.generate_content_stream()
        # and handle partial messages and tool call assembly.
        
        # For a simple generator:
        def gen():
            yield chat_response
        return gen()


    async def astream_chat_with_tools(
        self, tools: List[BaseTool], user_msg: Optional[str] = None, chat_history: Optional[List[ChatMessage]] = None, verbose: bool = False, allow_parallel_tool_calls: bool = False, **kwargs: Any
    ) -> ChatResponseGen: # Corrected return type annotation
        # For now, just a wrapper around the synchronous version.
        # TODO: Implement true async streaming version.
        logger.info("astream_chat_with_tools called, forwarding to synchronous stream_chat_with_tools.")
        
        # The synchronous stream_chat_with_tools returns a generator.
        # We need to adapt this to an async generator.
        sync_gen = self.stream_chat_with_tools(tools, user_msg, chat_history, verbose, allow_parallel_tool_calls, **kwargs)
        
        async def async_gen_wrapper():
            for item in sync_gen:
                yield item
        
        return async_gen_wrapper()

    def _prepare_chat_with_tools(
        self, 
        tools: Sequence[BaseTool], 
        user_msg: Optional[str] = None, 
        chat_history: Optional[List[ChatMessage]] = None, 
        verbose: bool = False, 
        allow_parallel_tool_calls: bool = False, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Prepare the chat with tools for the Gemini model.
        This method is required by FunctionCallingLLM base class.
        """
        # Convert tools to Gemini format
        gemini_tools_declarations = [
            self._llamaindex_tool_to_gemini_function_declaration(tool) for tool in tools
        ]
        gemini_sdk_tool = genai.types.Tool(function_declarations=gemini_tools_declarations)
        
        # Prepare messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        if user_msg:
            if isinstance(user_msg, str):
                messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
            else:
                messages.append(user_msg)
        
        return {
            "messages": messages,
            "tools": [gemini_sdk_tool] if gemini_tools_declarations else [],
            "allow_parallel_tool_calls": allow_parallel_tool_calls,
            "verbose": verbose,
        }

if __name__ == '__main__':
    # Example Usage (requires GOOGLE_API_KEY to be set in environment or passed)
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("GOOGLE_API_KEY not found. Please set it in your .env file.")
    else:
        # Test chat
        print("Testing Chat LLM...")
        llm = GeminiCustomLLM(model_name="gemini-2.0-flash", api_key=google_api_key, system_prompt="You are a witty pirate.")
        
        # Sync chat
        response = llm.chat([
            ChatMessage(role=MessageRole.USER, content="Ahoy! What be treasure?"),
        ])
        print(f"Assistant (sync chat): {response.message.content}")

        # Async chat
        async def test_async_chat():
            response_async = await llm.achat([
                ChatMessage(role=MessageRole.USER, content="Ahoy! How do ye find treasure, async style?"),
            ])
            print(f"Assistant (async chat): {response_async.message.content}")
        asyncio.run(test_async_chat())

        # Sync completion
        print("\nTesting Completion LLM...")
        completion_response = llm.complete("Write a short poem about the sea.")
        print(f"Assistant (sync completion): {completion_response.text}")

        # Async completion
        async def test_async_completion():
            completion_response_async = await llm.acomplete("Write a haiku about a ship, async.")
            print(f"Assistant (async completion): {completion_response_async.text}")
        asyncio.run(test_async_completion())
        
        # Test streaming chat
        print("\nTesting Streaming Chat LLM...")
        stream_response = llm.stream_chat([
            ChatMessage(role=MessageRole.USER, content="Tell me a short pirate joke, stream it!")
        ])
        print("Assistant (streaming chat): ", end="")
        for chunk in stream_response:
            print(chunk.delta, end="", flush=True)
        print("\n")

        # Test async streaming chat
        async def test_async_stream_chat():
            print("Assistant (async streaming chat): ", end="")
            async_stream_response = await llm.astream_chat([
                ChatMessage(role=MessageRole.USER, content="Another short pirate joke, async stream!")
            ])
            async for chunk in async_stream_response:
                print(chunk.delta, end="", flush=True)
            print("\n")
        asyncio.run(test_async_stream_chat())

        print("\nAll tests complete.") 