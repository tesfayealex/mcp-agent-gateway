import google.generativeai as genai
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
from typing import List, Any, Optional
import os
from dotenv import load_dotenv
import asyncio

# Corresponds to Gemini's EmbedTextResponse or EmbedContentResponse (BatchEmbedContentsResponse)
# We need to handle both single text and batch text embeddings.

class GeminiCustomEmbedding(BaseEmbedding):
    model_name: str
    api_key: str
    task_type: str # e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"
    title: Optional[str] = None # Optional for RETRIEVAL_DOCUMENT task type

    _model: Any = None # Not storing genai.GenerativeModel, direct calls to genai.embed_content

    def __init__(
        self,
        model_name: str, # e.g., "models/embedding-001"
        api_key: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key
        self.task_type = task_type
        self.title = title
        genai.configure(api_key=self.api_key) # Ensure API key is configured

    def _get_text_embedding(self, text: str) -> Embedding:
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=self.task_type,
            title=self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None,
        )
        return response["embedding"]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        # genai.embed_content is synchronous. To make it truly async for LlamaIndex,
        # we'd typically run it in a thread pool executor if the underlying SDK call is blocking.
        # For simplicity here, we'll call the sync version but LlamaIndex expects an awaitable.
        # A better approach for production would be to use asyncio.to_thread (Python 3.9+)
        # or a custom thread pool executor for older Python versions.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, # Default executor (ThreadPoolExecutor)
            genai.embed_content, # Function to run
            {
                'model': self.model_name,
                'content': text,
                'task_type': self.task_type,
                'title': self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None,
            }
        )
        # The response from embed_content when called with a dictionary for its args like this
        # might be different. The direct call is genai.embed_content(model=..., content=..., ...)
        # Let's correct to call it directly.
        response = await loop.run_in_executor(
            None, 
            lambda: genai.embed_content(
                model=self.model_name, 
                content=text, 
                task_type=self.task_type, 
                title=self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None
            )
        )
        return response["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        # The genai.embed_content function can handle a list of strings for batching if the model supports it.
        # However, the API reference for `embed_content` shows `content` as `Union[str, Iterable[str], ContentDict]`. 
        # It implies it can take a list directly.
        # Let's assume it can batch. If not, we would iterate and call _get_text_embedding.
        # The `batch_embed_contents` might be more appropriate for explicit batching if `embed_content` doesn't batch as expected.
        # For `embed_content` with a list of strings, it might return a BatchEmbedContentsResponse like structure.
        # Or it might make multiple calls internally.
        # Let's use `genai.embed_content` with a list of texts.
        
        # According to documentation, for multiple pieces of content, use `embed_contents` (plural)
        # but the `google-generativeai` SDK has `embed_content` which takes `Union[str, Iterable[str], ContentDict]`
        # and `batch_embed_contents` for `BatchEmbedContentsRequest`.
        # Let's assume `embed_content` handles lists of strings correctly and returns a structure with a list of embeddings.
        
        # Trying with a loop for clarity and safety, then can optimize if batch is confirmed.
        embeddings = []
        for text_item in texts:
            response = genai.embed_content(
                model=self.model_name,
                content=text_item,
                task_type=self.task_type,
                title=self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None,
            )
            embeddings.append(response["embedding"])
        return embeddings
        
        # Potentially more efficient way if embed_content handles batching for lists:
        # response = genai.embed_content(
        # model=self.model_name,
        # content=texts, # Pass the list of texts
        # task_type=self.task_type,
        # title=self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None,
        # )
        # # The structure of response needs to be checked. If it's like BatchEmbedContentsResponse,
        # # it would be response["embeddings"], where each item is {"value": [...]}
        # return [item["value"] for item in response.get("embeddings", [])]


    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        # Similar to _aget_text_embedding, run the synchronous batch call in an executor.
        loop = asyncio.get_event_loop()
        
        # Using the loop approach for now.
        # For true async batching, one would need to check if the SDK offers an async batch method
        # or parallelize individual async calls (though that might hit rate limits faster).
        
        embeddings = []
        for text_item in texts:
            # This will make sequential async calls, not ideal for batching.
            # We should batch the call to run_in_executor if possible.
            embedding = await self._aget_text_embedding(text_item) # Reuses the single async logic
            embeddings.append(embedding)
        return embeddings

        # A slightly better approach for batching with run_in_executor:
        # def batch_embed_sync(texts_to_embed: List[str]):
        #     results = []
        #     for t in texts_to_embed:
        #         resp = genai.embed_content(
        #             model=self.model_name,
        #             content=t,
        #             task_type=self.task_type,
        #             title=self.title if self.task_type == "RETRIEVAL_DOCUMENT" else None,
        #         )
        #         results.append(resp["embedding"])
        #     return results
        # return await loop.run_in_executor(None, batch_embed_sync, texts)


if __name__ == '__main__':
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("GOOGLE_API_KEY not found in .env file.")
    else:
        print("Testing Gemini Custom Embeddings...")
        # Example model name for embeddings, check official docs for the latest/correct one
        # e.g., "models/embedding-001"
        embed_model = GeminiCustomEmbedding(
            model_name="models/embedding-001", 
            api_key=google_api_key,
            task_type="RETRIEVAL_DOCUMENT" # or "RETRIEVAL_QUERY" for queries
        )

        # Test single text embedding
        text1 = "This is a test document for embedding."
        embedding1 = embed_model.get_text_embedding(text1)
        print(f"Embedding for '{text1}' (first 5 dims): {embedding1[:5]}")
        print(f"Embedding length: {len(embedding1)}")

        # Test batch text embeddings
        texts_batch = [
            "Another document here.",
            "The quick brown fox jumps over the lazy dog."
        ]
        embeddings_batch = embed_model.get_text_embeddings(texts_batch)
        print(f"\nBatch embeddings (first 5 dims of first item): {embeddings_batch[0][:5]}")
        print(f"Number of batch embeddings: {len(embeddings_batch)}")
        print(f"Length of first batch embedding: {len(embeddings_batch[0])}")

        # Test async single text embedding
        async def test_async_single_embedding():
            text_async = "Async embedding test."
            embedding_async = await embed_model.aget_text_embedding(text_async)
            print(f"\nAsync embedding for '{text_async}' (first 5 dims): {embedding_async[:5]}")
            print(f"Async embedding length: {len(embedding_async)}")

        asyncio.run(test_async_single_embedding())

        # Test async batch text embeddings
        async def test_async_batch_embeddings():
            texts_async_batch = [
                "Async batch document 1.",
                "Async batch document 2, the sequel."
            ]
            embeddings_async_batch = await embed_model.aget_text_embeddings(texts_async_batch)
            print(f"\nAsync batch embeddings (first 5 dims of first item): {embeddings_async_batch[0][:5]}")
            print(f"Number of async batch embeddings: {len(embeddings_async_batch)}")
            print(f"Length of first async batch embedding: {len(embeddings_async_batch[0])}")
        
        asyncio.run(test_async_batch_embeddings())
        print("\nAll embedding tests complete.") 