from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.query_engine import BaseQueryEngine
# from .custom_embedder import GeminiCustomEmbedding # If run as part of a package
# from .custom_llm import GeminiCustomLLM # If run as partof a package
from custom_embedder import GeminiCustomEmbedding # If run directly for testing
from custom_llm import GeminiCustomLLM # If run directly for testing

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PERSIST_DIR = "./storage" # Relative to this file, or choose a more robust path

def create_rag_query_engine(
    knowledge_base_path: str,
    custom_embedding_model: GeminiCustomEmbedding,
    custom_llm: Optional[GeminiCustomLLM] = None, # LLM for synthesis
    persist_dir: Optional[str] = None, # Optional: directory to persist and load the index
) -> BaseQueryEngine:
    """
    Creates or loads a RAG query engine using the specified knowledge base and custom embedding model.

    Args:
        knowledge_base_path: Path to the directory containing documents for the knowledge base.
        custom_embedding_model: An instance of GeminiCustomEmbedding.
        custom_llm: An instance of GeminiCustomLLM for the query engine's synthesizer.
                    If None, the global LlamaIndex Settings.llm will be used.
        persist_dir: If provided, the index will be persisted to this directory after creation
                     and loaded from this directory if it already exists.

    Returns:
        A LlamaIndex BaseQueryEngine.
    """
    logger.info(f"Initializing RAG query engine with knowledge base: {knowledge_base_path}")
    logger.info(f"Using embedding model: {custom_embedding_model.model_name}")
    if custom_llm:
        logger.info(f"Using LLM for synthesis: {custom_llm.model_name}")
        Settings.llm = custom_llm # Set for this context if provided
    else:
        logger.info(f"Using global Settings.llm for synthesis: {Settings.llm}")
        if not Settings.llm:
            logger.warning("Global Settings.llm is not set. Query engine might fail at synthesis.")

    Settings.embed_model = custom_embedding_model # Set for this context

    if persist_dir and os.path.exists(persist_dir):
        try:
            logger.info(f"Loading existing index from: {persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            # When loading, embed_model and llm are taken from global Settings or passed if different
            index = load_index_from_storage(
                storage_context,
                embed_model=custom_embedding_model, # Ensure correct embed model is used
            )
            logger.info("Successfully loaded index from storage.")
        except Exception as e:
            logger.warning(f"Failed to load index from {persist_dir}: {e}. Rebuilding index.")
            index = _build_index(knowledge_base_path, custom_embedding_model, persist_dir)
    else:
        index = _build_index(knowledge_base_path, custom_embedding_model, persist_dir)

    # Configure the query engine. Can specify similarity_top_k, response_synthesizer, etc.
    query_engine = index.as_query_engine(
        llm=custom_llm if custom_llm else Settings.llm, # Ensure LLM is passed for synthesis
        # similarity_top_k=3 # Example
    )
    logger.info("RAG query engine created successfully.")
    return query_engine

def _build_index(
    knowledge_base_path: str,
    custom_embedding_model: GeminiCustomEmbedding,
    persist_dir: Optional[str] = None,
) -> VectorStoreIndex:
    logger.info(f"Building new index from documents in: {knowledge_base_path}")
    if not os.path.exists(knowledge_base_path) or not os.listdir(knowledge_base_path):
        logger.error(f"Knowledge base path {knowledge_base_path} is empty or does not exist.")
        raise FileNotFoundError(f"No documents found in {knowledge_base_path}")

    documents = SimpleDirectoryReader(input_dir=knowledge_base_path).load_data()
    if not documents:
        logger.error(f"No documents were loaded from {knowledge_base_path}.")
        raise ValueError(f"No documents loaded from {knowledge_base_path}")
    
    logger.info(f"Loaded {len(documents)} document(s).")

    # Explicitly pass the embed_model during index construction
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=custom_embedding_model,
        show_progress=True
    )
    logger.info("Index construction complete.")

    if persist_dir:
        logger.info(f"Persisting index to: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info("Index persisted successfully.")
    return index

if __name__ == '__main__':
    # This example assumes .env is in the parent directory of dev_assistant_agent_py
    # and mock_knowledge_base is also in the parent directory.
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    knowledge_base_relative_path = os.getenv("KNOWLEDGE_BASE_PATH", "../mock_knowledge_base") # Default from plan
    kb_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', knowledge_base_relative_path.lstrip("./")))
    
    # Create a dummy knowledge base if it doesn't exist for testing
    if not os.path.exists(kb_full_path):
        os.makedirs(kb_full_path, exist_ok=True)
        with open(os.path.join(kb_full_path, "sample_doc.txt"), "w") as f:
            f.write("This is a sample document in the knowledge base. Gemini is a multimodal AI model by Google.")
        with open(os.path.join(kb_full_path, "another_doc.txt"), "w") as f:
            f.write("LlamaIndex is a framework for building RAG applications. MCP stands for Model Context Protocol.")
        logger.info(f"Created dummy knowledge base at: {kb_full_path}")

    storage_persist_dir = os.path.join(os.path.dirname(__file__), "test_rag_storage")
    logger.info(f"Using storage persist directory: {storage_persist_dir}")

    load_dotenv(dotenv_path=dotenv_path)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    chat_model_name = os.getenv("GEMINI_CHAT_MODEL_NAME", "gemini-1.5-flash-latest")
    embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/embedding-001")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found. Please set it in your .env file located at project root.")
    else:
        logger.info("Initializing custom models for RAG setup test...")
        my_gemini_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name,
            api_key=google_api_key,
            task_type="RETRIEVAL_DOCUMENT" # Use RETRIEVAL_DOCUMENT for indexing
        )
        my_gemini_llm = GeminiCustomLLM(
            model_name=chat_model_name,
            api_key=google_api_key
        )

        logger.info(f"Knowledge base path for test: {kb_full_path}")

        # Create query engine (this will build and persist the index the first time)
        query_engine = create_rag_query_engine(
            knowledge_base_path=kb_full_path,
            custom_embedding_model=my_gemini_embedder,
            custom_llm=my_gemini_llm,
            persist_dir=storage_persist_dir
        )

        # Test a query - ensure embedder for query is appropriate (e.g. RETRIEVAL_QUERY)
        # For simplicity, current embedder is doc type. For real use, might need a query-specific one.
        # We can set a query-specific task_type for the query engine's embedder if needed,
        # or the current embedder is general enough.
        
        # query_text = "What is Gemini?"
        # logger.info(f"Querying: {query_text}")
        # response = query_engine.query(query_text)
        # logger.info(f"Response: {response}")
        # logger.info(f"Source nodes: {response.source_nodes}")

        # # Test another query to see if it loads from storage (if run again)
        # query_text_2 = "Tell me about LlamaIndex."
        # logger.info(f"Querying: {query_text_2}")
        # # If testing loading, re-create the engine (it will try to load from persist_dir)
        # query_engine_loaded = create_rag_query_engine(
        #     knowledge_base_path=kb_full_path, # Still needed for potential rebuild
        #     custom_embedding_model=my_gemini_embedder,
        #     custom_llm=my_gemini_llm,
        #     persist_dir=storage_persist_dir
        # )
        # response_2 = query_engine_loaded.query(query_text_2)
        # logger.info(f"Response 2: {response_2}")

        # A more robust test: change the task type for query
        my_gemini_query_embedder = GeminiCustomEmbedding(
            model_name=embedding_model_name,
            api_key=google_api_key,
            task_type="RETRIEVAL_QUERY"
        )
        
        # Update Settings for the query phase if needed, or pass embedder to query engine if it supports it.
        # LlamaIndex query engines typically use Settings.embed_model at query time if not overridden.
        # So, for a query, we should ensure Settings.embed_model is set to the query embedder.
        
        query_text_3 = "What is MCP?"
        logger.info(f"Querying with specific query embedder setup: {query_text_3}")
        
        # Temporarily set the global embed_model for query phase
        original_embed_model = Settings.embed_model
        Settings.embed_model = my_gemini_query_embedder
        
        response_3 = query_engine.query(query_text_3)
        logger.info(f"Response 3: {response_3}")
        logger.info(f"Source nodes 3: {str(response_3.source_nodes)[:500]}...") # Print snippet

        # Restore original embed_model if it was changed
        Settings.embed_model = original_embed_model

        logger.info("RAG setup test completed.")
        logger.info(f"If you run this again, it should load the index from {storage_persist_dir} unless the dir is removed.")

from typing import Optional # Added for Python < 3.9 compatibility for Optional in function signature 