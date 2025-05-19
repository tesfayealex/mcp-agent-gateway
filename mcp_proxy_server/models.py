from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ManagedServerInfo(BaseModel):
    name: str
    status: str # e.g., "connected", "disconnected", "error", "disabled"
    connection_type: str # e.g., "stdio", "url"
    config_enabled: bool = Field(default=True, description="Whether the server is enabled in the configuration")

class ToolParameterSchema(BaseModel):
    name: str
    type_hint: str = Field(..., description="Python type hint for the parameter")
    required: bool
    description: Optional[str] = None
    default: Optional[Any] = None

class DownstreamToolSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: List[ToolParameterSchema] = Field(default_factory=list)
    # mcp_server_name: str # Context provided by the call to get_server_tools

class ListDownstreamToolsResponse(BaseModel):
    tools: List[DownstreamToolSchema]

class ToolCallResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str 