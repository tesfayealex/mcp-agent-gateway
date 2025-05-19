import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, validator, Field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class TestToolConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class StdioConfig(BaseModel):
    command: str
    args: List[str] = Field(default_factory=list)
    env_vars_to_pass: Dict[str, str] = Field(default_factory=dict) # subprocess_env_var_name: manager_env_var_name

class UrlConfig(BaseModel):
    base_url: str

class AuthConfig(BaseModel):
    type: str # e.g., "bearer_token"
    token_env_var: Optional[str] = None # manager_env_var_name for the token

class ServerConfig(BaseModel):
    name: str
    enabled: bool = True
    connection_type: str # "stdio" or "url"
    stdio_config: Optional[StdioConfig] = None
    url_config: Optional[UrlConfig] = None
    authentication: Optional[AuthConfig] = None
    test_tool: TestToolConfig
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: int = 5 # Delay between reconnection attempts

    @validator('connection_type')
    def connection_type_must_be_valid(cls, v):
        if v not in ['stdio', 'url']:
            raise ValueError('connection_type must be "stdio" or "url"')
        return v

    @validator('stdio_config', always=True)
    def check_stdio_config(cls, v, values):
        if values.get('connection_type') == 'stdio' and v is None:
            raise ValueError('stdio_config is required for stdio connection_type')
        return v

    @validator('url_config', always=True)
    def check_url_config(cls, v, values):
        if values.get('connection_type') == 'url' and v is None:
            raise ValueError('url_config is required for url connection_type')
        return v

    @validator('authentication', always=True)
    def check_auth_config(cls, v, values):
        if values.get('connection_type') == 'url' and v and not v.token_env_var:
            logger.warning(f"Authentication configured for server '{values.get('name')}' but no token_env_var specified.")
        return v

def load_and_resolve_env_vars(config_data_list: List[Dict[str, Any]]) -> List[ServerConfig]:
    """Loads configurations and resolves environment variables."""
    load_dotenv() # Load from .env file in the current working directory or parent dirs
    
    resolved_configs = []
    for config_data in config_data_list:
        # Resolve stdio_config.env_vars_to_pass
        if config_data.get('connection_type') == 'stdio' and config_data.get('stdio_config'):
            stdio_env_vars = config_data['stdio_config'].get('env_vars_to_pass', {})
            resolved_stdio_env = {}
            for process_var, manager_var_name in stdio_env_vars.items():
                value = os.getenv(manager_var_name)
                if value is None:
                    logger.warning(f"Environment variable '{manager_var_name}' not found for server '{config_data.get('name')}' stdio config.")
                resolved_stdio_env[process_var] = value
            # Replace the original env_vars_to_pass with the actual values (or None if not found)
            # The StdioTransport in fastmcp expects the actual values for its 'env' param.
            # We will pass this resolved dict to ServerHandler and it will use it.
            # For Pydantic model, we still store the *names* of manager env vars.
            # So this resolution happens *after* Pydantic validation of structure, but *before* ServerConfig instantiation for runtime use.
            # Let's adjust this logic: ServerConfig will store the *names*, and ServerHandler will resolve them at connection time.
            pass # No direct modification here, resolution happens in ServerHandler

        # Resolve authentication.token_env_var
        # Similar to above, resolution should happen in ServerHandler when establishing connection.
        if config_data.get('authentication') and config_data['authentication'].get('token_env_var'):
            token_env_name = config_data['authentication']['token_env_var']
            # token_value = os.getenv(token_env_name)
            # if token_value is None:
            # logger.warning(f"Authentication token environment variable '{token_env_name}' not found for server '{config_data.get('name')}.")
            # Store the actual token in a runtime field if needed, or let ServerHandler resolve it.
            pass 

        try:
            server_conf = ServerConfig(**config_data)
            resolved_configs.append(server_conf)
        except Exception as e:
            logger.error(f"Error validating configuration for server '{config_data.get('name', 'Unknown')}': {e}")
            # Decide if you want to skip invalid configs or raise an error
            # For now, we'll skip and log.
            continue 
            
    return resolved_configs

def load_configs(config_path: str = "config.json") -> List[ServerConfig]:
    """Loads MCP server configurations from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            raw_configs = json.load(f)
        if not isinstance(raw_configs, list):
            logger.error(f"Configuration file {config_path} must contain a JSON list.")
            return []
        return load_and_resolve_env_vars(raw_configs)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configs: {e}")
        return []

if __name__ == '__main__':
    # Basic logging setup for testing config_loader directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy .env file for testing
    with open(".env", "w") as f:
        f.write("LOCAL_DOCKER_GITHUB_PAT=dummy_pat_from_env\n")
        f.write("REMOTE_DEV_MCP_API_TOKEN=dummy_token_from_env\n")

    # Create a dummy config.json for testing
    dummy_config = [
        {
            "name": "Local Docker MCP Test",
            "enabled": True,
            "connection_type": "stdio",
            "stdio_config": {
                "command": "docker",
                "args": ["run", "-i", "--rm", "-e", "PAT_VAR=actual_pat", "some_image"],
                "env_vars_to_pass": {"PAT_VAR": "LOCAL_DOCKER_GITHUB_PAT"} # subprocess var: manager .env var
            },
            "test_tool": {"name": "get_me", "params": {}},
            "max_reconnect_attempts": 2,
            "reconnect_delay_seconds": 3
        },
        {
            "name": "Remote Dev MCP Test",
            "enabled": False,
            "connection_type": "url",
            "url_config": {"base_url": "http://localhost:8080"},
            "authentication": {"type": "bearer_token", "token_env_var": "REMOTE_DEV_MCP_API_TOKEN"},
            "test_tool": {"name": "list_tools", "params": {}},
        }
    ]
    with open("config.json", "w") as f:
        json.dump(dummy_config, f, indent=2)

    logger.info("Loading configurations...")
    configs = load_configs("config.json")

    if configs:
        logger.info(f"Successfully loaded {len(configs)} server configurations:")
        for cfg in configs:
            logger.info(f"  Name: {cfg.name}, Enabled: {cfg.enabled}, Type: {cfg.connection_type}")
            if cfg.stdio_config:
                logger.info(f"    STDIO Command: {cfg.stdio_config.command}")
                logger.info(f"    STDIO Env Needs Resolving: {cfg.stdio_config.env_vars_to_pass}")
            if cfg.url_config:
                logger.info(f"    URL: {cfg.url_config.base_url}")
            if cfg.authentication and cfg.authentication.token_env_var:
                logger.info(f"    Auth Token Env Var: {cfg.authentication.token_env_var} (Needs resolving by handler)")
            logger.info(f"    Test tool: {cfg.test_tool.name}")
    else:
        logger.error("No configurations were loaded.")

    # Clean up dummy files
    # os.remove(".env")
    # os.remove("config.json")
    logger.info("Config loader test finished. Remember to have a real .env and config.json for the main app.") 