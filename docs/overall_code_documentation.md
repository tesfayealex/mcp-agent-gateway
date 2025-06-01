# MCP Agent Gateway - Overall Code Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Configuration Management](#configuration-management)
- [API Specifications](#api-specifications)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Deployment Guide](#deployment-guide)
- [Development Guide](#development-guide)

## System Overview

The MCP Agent Gateway is a sophisticated, modular system designed to facilitate intelligent agent interactions with multiple external services through the Method Call Protocol (MCP). The system serves as a centralized gateway that manages connections to various downstream MCP servers while providing agents with a unified interface for tool discovery and execution.

### Key Objectives

- **Unified Tool Access**: Provide a single endpoint for agents to access multiple MCP servers
- **Service Abstraction**: Abstract away the complexity of individual MCP server protocols
- **Intelligent Assistance**: Enable development assistance through RAG-powered knowledge retrieval
- **Scalable Architecture**: Support dynamic addition and removal of MCP services
- **Robust Communication**: Ensure reliable communication between agents and services

### Core Technologies

- **FastAPI**: Web framework for the proxy server
- **FastMCP**: MCP protocol implementation
- **LlamaIndex**: RAG pipeline and agent framework
- **Google Gemini**: LLM and embedding services
- **Docker**: Containerization for MCP servers
- **Pydantic**: Data validation and configuration management

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP Agent Gateway System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                    ┌─────────────────────────────────┐ │
│  │                 │                    │                                 │ │
│  │ Dev Assistant   │◄──────────────────►│     MCP Proxy Server            │ │
│  │ Agent           │   MCP Protocol     │     (FastAPI + FastMCP)         │ │
│  │                 │                    │                                 │ │
│  └─────────────────┘                    └─────────────────┬───────────────┘ │
│          │                                               │                 │
│          │                                               │                 │
│          ▼                                               ▼                 │
│  ┌─────────────────┐                    ┌─────────────────────────────────┐ │
│  │                 │                    │                                 │ │
│  │ RAG Pipeline    │                    │     MCP Connection Manager      │ │
│  │ - Gemini LLM    │                    │     - Server Discovery          │ │
│  │ - Embeddings    │                    │     - Connection Pooling        │ │
│  │ - Vector Store  │                    │     - Health Monitoring         │ │
│  │                 │                    │                                 │ │
│  └─────────────────┘                    └─────────────────┬───────────────┘ │
│                                                           │                 │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────────┤ │
│  │                    Downstream MCP Servers                              │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │   GitHub    │  │ Filesystem  │  │   Custom    │  │   Future    │   │ │
│  │  │     MCP     │  │     MCP     │  │     MCP     │  │     MCP     │   │ │
│  │  │   Server    │  │   Server    │  │   Server    │  │   Servers   │   │ │
│  │  │             │  │             │  │             │  │             │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
Client Request → MCP Proxy Server → MCP Manager → Server Handler → MCP Server
                      ↓                    ↓              ↓
                 Tool Discovery       Connection       Tool Execution
                      ↓                    ↓              ↓
                Response Routing ← Result Processing ← Response
```

## Core Components

### 1. MCP Proxy Server (`mcp_proxy_server/`)

The central orchestrator that provides the main API endpoints for the system.

#### Key Files:

**`proxy_server.py`** - Main server implementation
```python
# Core server setup
proxy_mcp = FastMCP("MCP-Proxy-Server")
app = FastAPI(lifespan=app_lifespan)

# Tool definitions
@proxy_mcp.tool()
async def list_managed_servers() -> ListManagedServersResponse:
    """Discover all available downstream MCP servers"""

@proxy_mcp.tool()
async def call_server_tool(
    server_name: str, 
    tool_name: str, 
    arguments: Dict[str, Any]
) -> ToolCallResult:
    """Proxy tool calls to downstream servers"""
```

**`models.py`** - Data models and schemas
```python
class ManagedServerInfo(BaseModel):
    name: str
    enabled: bool
    connected: bool
    connection_type: str
    last_health_check: Optional[datetime]

class ToolCallResult(BaseModel):
    success: bool
    result: Optional[Any]
    error: Optional[str]
```

#### Responsibilities:
- Expose unified MCP interface
- Route tool calls to appropriate servers
- Manage server lifecycle events
- Provide discovery and status APIs
- Handle authentication and authorization

### 2. MCP Connection Manager (`mcp_manager/`)

Manages the lifecycle and connectivity of downstream MCP servers.

#### Key Files:

**`connection_manager.py`** - Central connection orchestrator
```python
class MCPConnectionManager:
    def __init__(self, server_configs: List[ServerConfig]):
        self.server_handlers: Dict[str, ServerHandler] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def connect_all_servers(self):
        """Establish connections to all enabled servers"""
    
    async def start_monitoring(self, check_interval_seconds: int = 60):
        """Begin health monitoring for all servers"""
```

**`server_handler.py`** - Individual server management
```python
class ServerHandler:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.client: Optional[Client] = None
        self.process: Optional[asyncio.subprocess.Process] = None
    
    async def connect(self) -> bool:
        """Establish connection based on config type (stdio/url)"""
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Execute tool on the managed server"""
```

**`config_loader.py`** - Configuration validation and loading
```python
class ServerConfig(BaseModel):
    name: str
    enabled: bool
    connection_type: Literal["stdio", "url"]
    stdio_config: Optional[StdioConfig]
    url_config: Optional[UrlConfig]
    authentication: Optional[AuthConfig]
    test_tool: TestToolConfig
```

#### Responsibilities:
- Load and validate server configurations
- Establish and maintain server connections
- Monitor server health and availability
- Handle reconnection logic
- Manage server-specific authentication

### 3. Dev Assistant Agent (`dev_assistant_agent_py/`)

An intelligent agent that combines MCP tool access with RAG-powered knowledge retrieval.

#### Key Files:

**`agent.py`** - Main agent implementation
```python
class DevAssistantAgent:
    def __init__(
        self,
        mcp_proxy_url: str,
        rag_query_engine: BaseQueryEngine,
        custom_llm: GeminiCustomLLM,
        tools: List[BaseTool]
    ):
        self.agent = FunctionAgent(
            tools=tools,
            llm=custom_llm,
            system_prompt=system_prompt,
            verbose=True
        )
    
    async def handle_message(self, user_query: str) -> str:
        """Process user queries using available tools and RAG"""
```

**`custom_llm.py`** - Gemini LLM integration
```python
class GeminiCustomLLM(CustomLLM):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.model = genai.GenerativeModel(model_name)
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Generate text completion using Gemini"""
```

**`custom_embedder.py`** - Gemini embedding integration
```python
class GeminiCustomEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "models/embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        self.task_type = task_type
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini"""
```

**`rag_setup.py`** - RAG pipeline orchestration
```python
def create_rag_query_engine(
    knowledge_base_path: str,
    custom_embedding_model: GeminiCustomEmbedding,
    custom_llm: Optional[GeminiCustomLLM] = None,
    persist_dir: Optional[str] = None,
) -> BaseQueryEngine:
    """Initialize complete RAG pipeline with persistence"""
```

#### Responsibilities:
- Integrate MCP tools with RAG capabilities
- Maintain conversation context
- Process natural language queries
- Coordinate between external tools and local knowledge
- Provide intelligent development assistance

## Data Flow

### 1. Agent Initialization Flow

```
Start → Load Environment → Initialize Models → Connect to Proxy → Discover Tools → Ready
   ↓           ↓               ↓                ↓               ↓          ↓
  .env → Google API → Gemini LLM/Embed → MCP Client → Tool List → Agent
```

### 2. Query Processing Flow

```
User Query → Agent → Tool Selection → Tool Execution → Response Synthesis
     ↓         ↓          ↓              ↓               ↓
 Natural → Function → MCP Proxy → Server Handler → Formatted
Language   Agent      Server       Connection      Response
```

### 3. MCP Tool Execution Flow

```
Tool Call → Proxy Server → Connection Manager → Server Handler → MCP Server
    ↓            ↓               ↓                   ↓             ↓
Arguments → Validation → Route Selection → Connection → Execution
    ↓            ↓               ↓                   ↓             ↓
Response ← Formatting ← Result Processing ← Raw Result ← Tool Result
```

### 4. RAG Query Flow

```
Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
  ↓        ↓            ↓              ↓                  ↓             ↓
Text → Vector → Similarity → Documents → Prompt + Context → Final Answer
```

## Configuration Management

### Environment Variables

The system uses environment-based configuration for flexibility and security:

```bash
# Core System Configuration
MCP_PROXY_URL="http://localhost:8100/proxy"
MCP_PROXY_HOST="0.0.0.0"
MCP_PROXY_PORT="8100"

# Google Gemini Configuration
GOOGLE_API_KEY="your_api_key_here"
GEMINI_CHAT_MODEL_NAME="gemini-1.5-flash-latest"
GEMINI_EMBEDDING_MODEL_NAME="models/embedding-001"

# RAG Configuration
KNOWLEDGE_BASE_PATH="../mock_knowledge_base"
RAG_STORAGE_PATH="./agent_rag_storage_prod"

# MCP Server Authentication
GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token"
LOCAL_DOCKER_GITHUB_PAT="github_token_for_docker"
REMOTE_DEV_MCP_API_TOKEN="remote_server_token"
```

### Server Configuration (`config.json`)

```json
[
  {
    "name": "GITHUB",
    "enabled": true,
    "connection_type": "stdio",
    "stdio_config": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "github-mcp-server:latest"
      ],
      "env_vars_to_pass": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "LOCAL_DOCKER_GITHUB_PAT"
      }
    },
    "test_tool": {
      "name": "get_me",
      "params": {}
    },
    "max_reconnect_attempts": 3,
    "reconnect_delay_seconds": 5
  },
  {
    "name": "filesystem",
    "enabled": true,
    "connection_type": "stdio",
    "stdio_config": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--mount", "type=bind,src=/path/to/data,dst=/projects",
        "mcp/filesystem",
        "/projects"
      ]
    },
    "test_tool": {
      "name": "list_files",
      "params": {}
    }
  }
]
```

### Configuration Validation

```python
class ServerConfig(BaseModel):
    name: str = Field(..., description="Unique server identifier")
    enabled: bool = Field(default=True, description="Whether server is active")
    connection_type: Literal["stdio", "url"] = Field(..., description="Connection method")
    stdio_config: Optional[StdioConfig] = Field(None, description="Subprocess configuration")
    url_config: Optional[UrlConfig] = Field(None, description="HTTP endpoint configuration")
    authentication: Optional[AuthConfig] = Field(None, description="Auth configuration")
    test_tool: TestToolConfig = Field(..., description="Health check tool")
    max_reconnect_attempts: int = Field(default=3, ge=0)
    reconnect_delay_seconds: int = Field(default=5, ge=1)
```

## API Specifications

### MCP Proxy Server API

#### Tool Discovery

**`list_managed_servers`**
```python
# Request: No parameters
# Response:
{
  "servers": [
    {
      "name": "GITHUB",
      "enabled": true,
      "connected": true,
      "connection_type": "stdio",
      "last_health_check": "2024-01-01T12:00:00Z",
      "available_tools": ["get_me", "create_repository", "list_issues"]
    }
  ]
}
```

**`get_server_tools`**
```python
# Request:
{
  "server_name": "GITHUB"
}

# Response:
{
  "server_name": "GITHUB",
  "tools": [
    {
      "name": "create_repository",
      "description": "Create a new GitHub repository",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "Repository name"},
          "description": {"type": "string", "description": "Repository description"},
          "private": {"type": "boolean", "default": false}
        },
        "required": ["name"]
      }
    }
  ]
}
```

#### Tool Execution

**`call_server_tool`**
```python
# Request:
{
  "server_name": "GITHUB",
  "tool_name": "create_repository",
  "arguments": {
    "name": "my-new-repo",
    "description": "A test repository",
    "private": false
  }
}

# Response:
{
  "success": true,
  "result": {
    "id": 123456789,
    "name": "my-new-repo",
    "full_name": "username/my-new-repo",
    "html_url": "https://github.com/username/my-new-repo"
  },
  "error": null
}
```

### Agent API

#### Message Handling

```python
# Input: Natural language query
query = "Create a new GitHub repository called 'test-project' with a description"

# Output: Structured response with tool execution results
response = {
  "message": "I've successfully created the repository 'test-project' for you.",
  "actions_taken": [
    {
      "tool": "GITHUB_create_repository",
      "parameters": {
        "name": "test-project",
        "description": "A test repository created by the assistant"
      },
      "result": "Repository created successfully"
    }
  ],
  "knowledge_used": [],
  "conversation_context": "Repository creation task completed"
}
```

## Security Considerations

### Authentication and Authorization

1. **API Key Management**
   ```python
   # Secure environment variable handling
   api_key = os.getenv("GOOGLE_API_KEY")
   if not api_key:
       raise ValueError("Missing required API key")
   ```

2. **Token Rotation**
   ```python
   # Support for dynamic token updates
   class AuthConfig(BaseModel):
       type: Literal["bearer_token", "api_key"]
       token_env_var: str
       refresh_interval: Optional[int] = None
   ```

3. **Access Control**
   ```python
   # Tool-level access control
   class ToolPermission(BaseModel):
       tool_name: str
       allowed_operations: List[str]
       rate_limit: Optional[int] = None
   ```

### Data Protection

1. **Environment Isolation**
   - Separate environment files for different deployment stages
   - Docker container isolation for MCP servers
   - Network segmentation for external services

2. **Logging Security**
   ```python
   # Sanitized logging to prevent credential exposure
   def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
       sensitive_keys = ['token', 'password', 'api_key', 'secret']
       return {k: '***' if any(s in k.lower() for s in sensitive_keys) 
               else v for k, v in data.items()}
   ```

3. **Input Validation**
   ```python
   # Comprehensive input sanitization
   class ToolCallRequest(BaseModel):
       server_name: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
       tool_name: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
       arguments: Dict[str, Any] = Field(default_factory=dict)
   ```

## Performance Optimization

### Connection Pooling

```python
class MCPConnectionManager:
    def __init__(self):
        self.connection_pool: Dict[str, List[Client]] = {}
        self.pool_size = 5
    
    async def get_connection(self, server_name: str) -> Client:
        """Get or create connection from pool"""
        if server_name in self.connection_pool:
            if self.connection_pool[server_name]:
                return self.connection_pool[server_name].pop()
        return await self._create_new_connection(server_name)
```

### Caching Strategies

```python
# Tool definition caching
@lru_cache(maxsize=128)
def get_server_tools(server_name: str) -> List[ToolDefinition]:
    """Cache tool definitions to reduce discovery calls"""

# RAG result caching
class QueryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
```

### Asynchronous Processing

```python
# Concurrent tool execution
async def execute_multiple_tools(tool_calls: List[ToolCall]) -> List[ToolResult]:
    tasks = [execute_single_tool(call) for call in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# Background health monitoring
async def monitor_server_health(self):
    while self.monitoring_enabled:
        await asyncio.gather(*[
            handler.health_check() 
            for handler in self.server_handlers.values()
        ])
        await asyncio.sleep(self.check_interval)
```

### Memory Management

```python
# RAG index optimization
class OptimizedRAGEngine:
    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.similarity_top_k = 5
    
    def optimize_index(self):
        """Periodic index optimization and cleanup"""
        self.index.storage_context.persist()
        self.clear_memory_cache()
```

## Deployment Guide

### Development Environment

```bash
# 1. Clone repository
git clone <repository-url>
cd mcp-agent-gateway

# 2. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.sample .env
# Edit .env with your configuration

# 5. Start proxy server
uvicorn mcp_proxy_server.proxy_server:proxy_mcp --host 0.0.0.0 --port 8100 --reload

# 6. Run agent (in separate terminal)
cd dev_assistant_agent_py
python run.py
```

### Production Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy
    ports:
      - "8100:8100"
    environment:
      - MCP_PROXY_HOST=0.0.0.0
      - MCP_PROXY_PORT=8100
    volumes:
      - ./config.json:/app/config.json
      - ./logs:/app/logs

  dev-assistant:
    build:
      context: .
      dockerfile: Dockerfile.agent
    depends_on:
      - mcp-proxy
    environment:
      - MCP_PROXY_URL=http://mcp-proxy:8100/proxy
    volumes:
      - ./knowledge_base:/app/knowledge_base
      - ./agent_storage:/app/agent_storage
```

### Health Monitoring

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "servers": await get_server_health_status()
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "active_connections": len(connection_manager.server_handlers),
        "total_requests": request_counter.value,
        "average_response_time": response_time_avg.value,
        "error_rate": error_rate.value
    }
```

## Development Guide

### Code Structure Guidelines

1. **Module Organization**
   ```
   component_name/
   ├── __init__.py          # Package initialization
   ├── main.py             # Entry point (if applicable)
   ├── models.py           # Data models and schemas
   ├── handlers/           # Business logic handlers
   ├── utils/              # Utility functions
   ├── tests/              # Unit tests
   └── README.md           # Component documentation
   ```

2. **Naming Conventions**
   - Classes: `PascalCase` (e.g., `MCPConnectionManager`)
   - Functions: `snake_case` (e.g., `connect_all_servers`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
   - Files: `snake_case` (e.g., `connection_manager.py`)

3. **Type Annotations**
   ```python
   from typing import List, Dict, Optional, Union, Any
   
   async def call_tool(
       self, 
       tool_name: str, 
       arguments: Dict[str, Any]
   ) -> Optional[Dict[str, Any]]:
       """Execute tool with type-safe parameters"""
   ```

### Testing Guidelines

```python
# Unit test example
import pytest
from unittest.mock import AsyncMock, patch

class TestMCPConnectionManager:
    @pytest.fixture
    async def manager(self):
        configs = [create_test_config("test_server")]
        return MCPConnectionManager(configs)
    
    @pytest.mark.asyncio
    async def test_connect_all_servers(self, manager):
        with patch.object(manager, '_connect_server') as mock_connect:
            mock_connect.return_value = True
            result = await manager.connect_all_servers()
            assert result is True
            mock_connect.assert_called_once()

# Integration test example
@pytest.mark.integration
async def test_end_to_end_tool_call():
    # Setup test environment
    proxy_server = await start_test_proxy()
    agent = await create_test_agent()
    
    try:
        # Execute test scenario
        response = await agent.handle_message("List my GitHub repositories")
        assert "repositories" in response.lower()
    finally:
        await cleanup_test_environment()
```

### Error Handling Patterns

```python
# Comprehensive error handling
class MCPError(Exception):
    """Base exception for MCP operations"""
    def __init__(self, message: str, server_name: Optional[str] = None):
        self.message = message
        self.server_name = server_name
        super().__init__(message)

class ConnectionError(MCPError):
    """Server connection failures"""
    pass

class ToolExecutionError(MCPError):
    """Tool execution failures"""
    def __init__(self, message: str, tool_name: str, **kwargs):
        self.tool_name = tool_name
        super().__init__(message, **kwargs)

# Error handling decorator
def handle_mcp_errors(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise MCPError(f"Operation failed: {str(e)}")
    return wrapper
```

### Logging Configuration

```python
# Structured logging setup
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'server_name'):
            log_data['server_name'] = record.server_name
        if hasattr(record, 'tool_name'):
            log_data['tool_name'] = record.tool_name
            
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_gateway.log")
    ]
)

# Usage in code
logger = logging.getLogger(__name__)
logger.info("Server connected", extra={
    "server_name": "GITHUB",
    "connection_type": "stdio"
})
```

This documentation provides a comprehensive overview of the MCP Agent Gateway system architecture, implementation details, and operational guidelines. For component-specific details, refer to the individual README files in each component directory.

---

*Last updated: January 2024*
*Version: 1.0.0* 