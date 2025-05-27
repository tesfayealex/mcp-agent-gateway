from google import genai
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
from typing import List, Any, Optional
from pydantic import Field, PrivateAttr
import os
from dotenv import load_dotenv
import asyncio

# Using the new Google genai client for embeddings
# Corresponds to the newer Gemini embedding API

class GeminiCustomEmbedding(BaseEmbedding):
    model_name: str = Field(description="The name of the Gemini embedding model to use")
    api_key: str = Field(description="The API key for Google Gemini", exclude=True)
    task_type: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings")
    title: Optional[str] = Field(default=None, description="Optional title for RETRIEVAL_DOCUMENT task type")

    _client: Any = PrivateAttr(default=None)  # Store the genai client

    def __init__(
        self,
        model_name: str = "gemini-embedding-exp-03-07", # Updated to use the newer model
        api_key: str = "",
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        **kwargs: Any,
    ):
        # Call super().__init__ with explicit field values
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            task_type=task_type,
            title=title,
            **kwargs
        )
        
        # Create the genai client after Pydantic initialization
        self._client = genai.Client(api_key=self.api_key)

    def _get_text_embedding(self, text: str) -> Embedding:
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return result.embeddings[0].values

    async def _aget_text_embedding(self, text: str) -> Embedding:
        # For async, we'll run the sync method in an executor since the new client doesn't have async methods yet
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self._client.models.embed_content(
                model=self.model_name, 
                contents=text
            )
        )
        return result.embeddings[0].values

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        # Process texts one by one for now - batch processing might be available later
        embeddings = []
        for text_item in texts:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text_item,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        # Process texts one by one using async single embedding method
        embeddings = []
        for text_item in texts:
            embedding = await self._aget_text_embedding(text_item)
            embeddings.append(embedding)
        return embeddings

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get embedding for a query string. Uses RETRIEVAL_QUERY task type."""
        # Note: The new API might handle task types differently
        # For now, we'll use the same method but could add config later
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=query,
        )
        return result.embeddings[0].values

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async version of getting embedding for a query string."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self._client.models.embed_content(
                model=self.model_name, 
                contents=query
            )
        )
        return result.embeddings[0].values

if __name__ == '__main__':
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("GOOGLE_API_KEY not found in .env file.")
    else:
        print("Testing Gemini Custom Embeddings...")
        # Using the new gemini-embedding model
        embed_model = GeminiCustomEmbedding(
            model_name="gemini-embedding-exp-03-07", 
            api_key=google_api_key,
            task_type="RETRIEVAL_DOCUMENT"
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