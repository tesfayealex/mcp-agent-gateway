import google.generativeai as genai
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager

from typing import Any, List, Sequence, Optional, AsyncIterator, Iterator

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

class GeminiCustomLLM(CustomLLM):
    model_name: str
    api_key: str
    temperature: float = 0.7
    top_p: float = 1.0
    # Add other generation config parameters as needed, e.g., max_output_tokens

    _model: Any = None
    _is_chat_model: bool = True # Assuming Gemini models are chat-like

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        # Add other params
    ):
        super().__init__(callback_manager=callback_manager)
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt # Store system prompt

        genai.configure(api_key=self.api_key)
        
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            # candidate_count=1, # Default is 1
            # max_output_tokens=...
        )

        safety_settings = [ # Adjust as needed
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]
        
        # For Gemini, system instructions are part of the model initialization or per-request
        # If a system prompt is provided, it's handled differently than in OpenAI models
        # It's passed to GenerativeModel or as part of the contents for generate_content
        
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=self.system_prompt if self.system_prompt else None
        )


    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,  # Example, adjust based on the specific Gemini model
            num_output=2048,      # Example, adjust
            is_chat_model=self._is_chat_model,
            model_name=self.model_name,
        )

    def _prepare_gemini_chat_history(self, messages: Sequence[ChatMessage]) -> List[genai.types.ContentDict]:
        history = []
        for msg in messages:
            role = _to_gemini_role(msg.role)
            # Skip system messages if they are handled by system_instruction at model init
            # or ensure they are formatted correctly if the model expects them in contents
            if msg.role == MessageRole.SYSTEM and self.system_prompt: 
                # If system_prompt is set at model level, skip here to avoid duplication/conflict
                # Or, if you want to allow overriding system_prompt per call, include it
                continue 
            history.append({'role': role, 'parts': [{'text': msg.content}]})
        return history

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        gemini_history = self._prepare_gemini_chat_history(messages)
        
        # If there's a system message and it wasn't set at model init,
        # or if we want to allow per-request system messages:
        current_system_prompt = self.system_prompt
        if messages and messages[0].role == MessageRole.SYSTEM and not self.system_prompt:
            current_system_prompt = messages[0].content # Use the first system message if no global one
            gemini_history = gemini_history[1:] # Don't send it as part of history if handled this way

        chat_session = self._model.start_chat(history=gemini_history[:-1] if len(gemini_history) > 1 else [])

        response = chat_session.send_message(
            content=gemini_history[-1]['parts'] if gemini_history else " ", # Send last message
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.text,
            ),
            raw=response,
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        gemini_history = self._prepare_gemini_chat_history(messages)
        
        current_system_prompt = self.system_prompt
        if messages and messages[0].role == MessageRole.SYSTEM and not self.system_prompt:
            current_system_prompt = messages[0].content
            # If we use system_instruction in the model, we might not need to remove it from history
            # Or we ensure it's the first part of contents sent.
            # For start_chat, history is prior turns. The system instruction is global to the GenerativeModel.
            # The last message in gemini_history will be the current user query.

        # For async, if start_chat doesn't have an async version, we might need to call generate_content_async
        # Let's assume start_chat + send_message handles async if underlying client is async,
        # or we directly use generate_content_async for stateless requests.

        # Simplification: using generate_content_async for stateless chat completion
        # This means history needs to be passed entirely each time if not using model.start_chat
        
        # Re-creating model instance with system_prompt if it's passed per-request and not global
        # This is not ideal. Better to use one model instance.
        # For `generate_content`, system_instruction can be part of `GenerationConfig` or top-level param
        # to `GenerativeModel`.
        # The current `self._model` is already configured with `system_instruction`.

        # If gemini_history includes a system message and system_prompt is also set,
        # `genai.GenerativeModel` will use `system_instruction` if provided at init.
        # `generate_content` can also take `system_instruction` in `GenerationConfig`.

        effective_history = gemini_history
        if self.system_prompt and effective_history and effective_history[0]['role'] == 'user' and messages[0].role == MessageRole.SYSTEM:
            # If system_prompt is set on model, and first message is also system, avoid duplication.
            # The current _prepare_gemini_chat_history already skips system messages if self.system_prompt is set.
            pass


        response = await self._model.generate_content_async(
            contents=effective_history, # Pass the whole history
            # generation_config can be passed here too if needed to override model's default
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.text,
            ),
            raw=response,
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        gemini_history = self._prepare_gemini_chat_history(messages)
        
        chat_session = self._model.start_chat(history=gemini_history[:-1] if len(gemini_history) > 1 else [])
        
        stream = chat_session.send_message(
            content=gemini_history[-1]['parts'] if gemini_history else " ",
            stream=True
        )

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in stream:
                delta = chunk.text
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=delta,
                    raw=chunk,
                )
        return gen()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        gemini_history = self._prepare_gemini_chat_history(messages)

        # Using generate_content_async for streaming
        stream = await self._model.generate_content_async(
            contents=gemini_history,
            stream=True
        )

        async def gen() -> ChatResponseGen:
            content = ""
            async for chunk in stream:
                delta = chunk.text
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    delta=delta,
                    raw=chunk,
                )
        # LlamaIndex expects an async generator here.
        # The return type hint `ChatResponseGen` actually means `AsyncIterator[ChatResponse]` for async
        # and `Iterator[ChatResponse]` for sync.
        return gen() # type: ignore


    # For FunctionAgent, it might also call complete/acomplete if it treats it as a completion task
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        # If system_prompt is set, it's part of self._model configuration.
        # For a simple completion, we can just send the prompt as user content.
        contents = [{'role': 'user', 'parts': [{'text': prompt}]}]
        # If self.system_prompt is available, and the model is NOT already configured with it,
        # we might prepend a system message to contents. However, self._model IS configured.

        response = self._model.generate_content(contents=contents)
        return CompletionResponse(text=response.text, raw=response)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        contents = [{'role': 'user', 'parts': [{'text': prompt}]}]
        response = await self._model.generate_content_async(contents=contents)
        return CompletionResponse(text=response.text, raw=response)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        contents = [{'role': 'user', 'parts': [{'text': prompt}]}]
        stream = self._model.generate_content(contents=contents, stream=True)
        
        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in stream:
                delta = chunk.text
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)
        return gen()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        contents = [{'role': 'user', 'parts': [{'text': prompt}]}]
        stream = await self._model.generate_content_async(contents=contents, stream=True)

        async def gen() -> CompletionResponseGen:
            text = ""
            async for chunk in stream:
                delta = chunk.text
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)
        return gen() # type: ignore

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
        llm = GeminiCustomLLM(model_name="gemini-1.5-flash-latest", api_key=google_api_key, system_prompt="You are a witty pirate.")
        
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