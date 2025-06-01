# RAG Pipeline Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Implementation Details](#implementation-details)
- [Configuration](#configuration)
- [Usage](#usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The RAG (Retrieval Augmented Generation) pipeline in the `mcp-agent-gateway` system is a sophisticated knowledge retrieval and generation system that enables the Dev Assistant Agent to answer queries using both external tools (via MCP) and local knowledge bases. The pipeline leverages Google's Gemini models for both embeddings and language generation, integrated through the LlamaIndex framework.

### Key Features

- **Multi-Modal Knowledge Access**: Combines local document search with external tool capabilities
- **Persistent Vector Storage**: Efficient indexing and retrieval with disk persistence
- **Google Gemini Integration**: Uses state-of-the-art embedding and language models
- **Asynchronous Processing**: Non-blocking operations for better performance
- **Flexible Configuration**: Environment-based configuration for different deployment scenarios

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dev Assistant Agent                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │   MCP Tools     │  │   RAG Pipeline   │  │  Conversation       │ │
│  │   Integration   │  │                  │  │  Management         │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                         RAG Components                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  Custom Gemini  │  │  Document       │  │  Vector Store       │ │
│  │  LLM Wrapper    │  │  Loader         │  │  Index              │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  Custom Gemini  │  │  Query Engine    │  │  Storage            │ │
│  │  Embeddings     │  │                  │  │  Persistence        │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Custom LLM Integration (`custom_llm.py`)

The `GeminiCustomLLM` class provides a bridge between Google's Gemini API and LlamaIndex's LLM interface.

**Key Features:**
- Supports multiple Gemini model variants (gemini-1.5-flash-latest, gemini-pro, etc.)
- Handles both streaming and non-streaming responses
- Comprehensive error handling and logging
- Token usage tracking and optimization
- Safety settings configuration

**Implementation Highlights:**
```python
class GeminiCustomLLM(CustomLLM):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # Initialization with comprehensive parameter validation
```

### 2. Custom Embedding Integration (`custom_embedder.py`)

The `GeminiCustomEmbedding` class implements document and query embeddings using Google's embedding models.

**Key Features:**
- Task-specific embedding optimization (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY)
- Batch processing for efficiency
- Configurable embedding dimensions
- Async and sync operation support

**Task Types:**
- `RETRIEVAL_DOCUMENT`: Optimized for document indexing
- `RETRIEVAL_QUERY`: Optimized for search queries
- `CLUSTERING`: For document clustering tasks
- `CLASSIFICATION`: For text classification

### 3. RAG Setup and Management (`rag_setup.py`)

The core RAG pipeline orchestrator that handles:

**Document Processing:**
```python
def create_rag_query_engine(
    knowledge_base_path: str,
    custom_embedding_model: GeminiCustomEmbedding,
    custom_llm: Optional[GeminiCustomLLM] = None,
    persist_dir: Optional[str] = None,
) -> BaseQueryEngine:
```

**Index Management:**
- Automatic index creation and persistence
- Efficient loading from existing storage
- Fallback rebuilding on corruption
- Progress tracking during indexing

### 4. Agent Integration (`agent.py`)

The `DevAssistantAgent` class integrates the RAG pipeline with MCP tools:

**RAG Tool Integration:**
```python
local_knowledge_tool = QueryEngineTool.from_defaults(
    query_engine=rag_query_engine,
    name="LocalKnowledgeBaseSearch",
    description="Search the local knowledge base for development-related information"
)
```

## Implementation Details

### Document Processing Pipeline

1. **Document Loading**
   ```python
   documents = SimpleDirectoryReader(input_dir=knowledge_base_path).load_data()
   ```
   - Supports multiple file formats (TXT, MD, PDF, etc.)
   - Recursive directory scanning
   - Metadata extraction and preservation

2. **Embedding Generation**
   ```python
   index = VectorStoreIndex.from_documents(
       documents,
       embed_model=custom_embedding_model,
       show_progress=True
   )
   ```
   - Document-optimized embeddings using `RETRIEVAL_DOCUMENT` task type
   - Batch processing for large document sets
   - Progress monitoring and logging

3. **Index Persistence**
   ```python
   index.storage_context.persist(persist_dir=persist_dir)
   ```
   - Efficient storage using LlamaIndex's storage context
   - Automatic directory creation
   - Metadata and index state preservation

### Query Processing Pipeline

1. **Query Embedding**
   - Uses `RETRIEVAL_QUERY` task type for optimal search performance
   - Dynamic task type switching based on query context

2. **Similarity Search**
   - Vector similarity computation
   - Configurable retrieval parameters (`similarity_top_k`)
   - Result ranking and filtering

3. **Response Generation**
   ```python
   query_engine = index.as_query_engine(
       llm=custom_llm if custom_llm else Settings.llm,
       similarity_top_k=3
   )
   ```
   - Context-aware response synthesis
   - Source citation and attribution
   - Conversation history integration

### Storage and Persistence

**Directory Structure:**
```
agent_rag_storage_prod/
├── docstore.json          # Document metadata
├── index_store.json       # Index configuration
├── vector_store.json      # Vector embeddings
└── graph_store.json       # Graph relationships
```

**Storage Features:**
- Incremental updates support
- Corruption detection and recovery
- Cross-platform compatibility
- Compression and optimization

## Configuration

### Environment Variables

```bash
# Core RAG Configuration
KNOWLEDGE_BASE_PATH="../mock_knowledge_base"
RAG_STORAGE_PATH="./agent_rag_storage_prod"

# Gemini API Configuration
GOOGLE_API_KEY="your_api_key_here"
GEMINI_CHAT_MODEL_NAME="gemini-1.5-flash-latest"
GEMINI_EMBEDDING_MODEL_NAME="models/embedding-001"

# Query Engine Settings
RAG_SIMILARITY_TOP_K=3
RAG_TEMPERATURE=0.1
RAG_MAX_TOKENS=4096
```

### Model Configuration

**LLM Settings:**
```python
custom_llm = GeminiCustomLLM(
    model_name="gemini-1.5-flash-latest",
    api_key=google_api_key,
    temperature=0.1,
    max_tokens=4096
)
```

**Embedding Settings:**
```python
embedding_model = GeminiCustomEmbedding(
    model_name="models/embedding-001",
    api_key=google_api_key,
    task_type="RETRIEVAL_DOCUMENT"  # or "RETRIEVAL_QUERY"
)
```

## Usage

### Basic RAG Query Engine Creation

```python
from rag_setup import create_rag_query_engine
from custom_llm import GeminiCustomLLM
from custom_embedder import GeminiCustomEmbedding

# Initialize models
llm = GeminiCustomLLM(api_key="your_key")
embedder = GeminiCustomEmbedding(api_key="your_key")

# Create query engine
query_engine = create_rag_query_engine(
    knowledge_base_path="./knowledge_base",
    custom_embedding_model=embedder,
    custom_llm=llm,
    persist_dir="./storage"
)

# Query the engine
response = query_engine.query("What is the MCP protocol?")
print(response.response)
```

### Integration with Dev Assistant Agent

```python
# The agent automatically integrates RAG as a tool
agent = await DevAssistantAgent.create(
    mcp_proxy_url="http://localhost:8100/proxy",
    rag_query_engine=query_engine,
    custom_llm=llm
)

# RAG is available as "LocalKnowledgeBaseSearch" tool
response = await agent.handle_message(
    "Search the knowledge base for information about Docker deployment"
)
```

### Manual Testing and Validation

```python
# Test RAG setup independently
if __name__ == '__main__':
    # Creates test knowledge base if needed
    # Builds or loads index
    # Performs sample queries
    # Validates storage and retrieval
```

## Best Practices

### 1. Knowledge Base Organization

**Directory Structure:**
```
knowledge_base/
├── documentation/
│   ├── api_docs/
│   ├── user_guides/
│   └── technical_specs/
├── examples/
│   ├── code_samples/
│   └── tutorials/
└── reference/
    ├── glossary/
    └── faqs/
```

**File Naming Conventions:**
- Use descriptive, searchable filenames
- Include version numbers for versioned documents
- Use consistent file extensions (.md, .txt, .pdf)

### 2. Embedding Optimization

**Task Type Selection:**
```python
# For indexing documents
document_embedder = GeminiCustomEmbedding(task_type="RETRIEVAL_DOCUMENT")

# For processing queries
query_embedder = GeminiCustomEmbedding(task_type="RETRIEVAL_QUERY")
```

**Batch Processing:**
- Process documents in batches for efficiency
- Monitor API rate limits
- Implement retry logic for failed embeddings

### 3. Query Engine Tuning

**Retrieval Parameters:**
```python
query_engine = index.as_query_engine(
    similarity_top_k=5,           # Number of retrieved documents
    response_mode="compact",      # Response synthesis mode
    streaming=True               # Enable streaming responses
)
```

**Performance Optimization:**
- Cache frequently accessed embeddings
- Use appropriate similarity thresholds
- Implement query result caching

### 4. Storage Management

**Persistence Strategy:**
- Regular index backups
- Incremental updates for large knowledge bases
- Storage cleanup and optimization

**Monitoring:**
- Track index size and performance
- Monitor query latency and accuracy
- Log embedding generation costs

## Troubleshooting

### Common Issues

#### 1. Index Loading Failures

**Symptoms:**
- "Failed to load index from storage" errors
- Corrupted storage warnings

**Solutions:**
```python
# Force index rebuild
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    
# Recreate with error handling
try:
    query_engine = create_rag_query_engine(...)
except Exception as e:
    logger.error(f"Index creation failed: {e}")
    # Implement fallback strategy
```

#### 2. Embedding API Errors

**Symptoms:**
- API quota exceeded
- Authentication failures
- Network timeouts

**Solutions:**
```python
# Implement retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_embedding(text):
    return embedding_model.get_text_embedding(text)

# Monitor API usage
def check_api_quota():
    # Implement quota checking logic
    pass
```

#### 3. Memory Issues

**Symptoms:**
- Out of memory errors during indexing
- Slow query performance

**Solutions:**
```python
# Batch document processing
def process_documents_in_batches(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        yield batch

# Optimize vector store
Settings.chunk_size = 512
Settings.chunk_overlap = 20
```

#### 4. Query Quality Issues

**Symptoms:**
- Irrelevant search results
- Poor response quality

**Solutions:**
```python
# Adjust retrieval parameters
query_engine = index.as_query_engine(
    similarity_top_k=10,          # Retrieve more candidates
    response_synthesizer=get_response_synthesizer(
        response_mode="tree_summarize",
        use_async=True
    )
)

# Implement query preprocessing
def preprocess_query(query):
    # Add context, expand abbreviations, etc.
    return enhanced_query
```

### Debugging and Logging

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

**Monitor Performance:**
```python
import time

def timed_query(query_engine, query):
    start_time = time.time()
    response = query_engine.query(query)
    elapsed_time = time.time() - start_time
    logger.info(f"Query took {elapsed_time:.2f} seconds")
    return response
```

### Performance Monitoring

**Key Metrics:**
- Query response time
- Index size and growth
- Embedding generation costs
- Memory usage patterns
- API rate limit usage

**Monitoring Tools:**
```python
def log_rag_metrics(query, response, elapsed_time):
    metrics = {
        "query_length": len(query),
        "response_length": len(response.response),
        "sources_count": len(response.source_nodes),
        "elapsed_time": elapsed_time
    }
    logger.info(f"RAG Metrics: {metrics}")
```

---

*This documentation provides a comprehensive overview of the RAG pipeline implementation. For specific code examples and detailed API documentation, refer to the individual component files in the `dev_assistant_agent_py` directory.* 