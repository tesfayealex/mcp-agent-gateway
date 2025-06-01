import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from fastmcp.client import StdioTransport, Client as FastMCPClient
import aiohttp

from .config_loader import ServerConfig

import fastmcp



logger = logging.getLogger(__name__)

class MCPClientWrapper:
    def __init__(self, server_config: ServerConfig):
        self.config: ServerConfig = server_config
        self.status: str = "disconnected"
        self._transport: Optional[StdioTransport] = None
        self._http_client: Optional[FastMCPClient] = None
        self._http_client_session: Optional[aiohttp.ClientSession] = None # For URL client if we manage session separately
        self.last_heartbeat: Optional[datetime] = None
        self.reconnect_attempts: int = 0

    def _resolve_env_var(self, env_var_name: Optional[str]) -> Optional[str]:
        if not env_var_name:
            return None
        value = os.getenv(env_var_name)
        if value is None:
            logger.warning(f"Server '{self.config.name}': Environment variable '{env_var_name}' not found.")
        return value

    def _get_stdio_env(self) -> Dict[str, str]:
        resolved_env = {}
        if self.config.stdio_config and self.config.stdio_config.env_vars_to_pass:
            # print("####################################")
            # print(f"Env vars to pass: {self.config.stdio_config.env_vars_to_pass}")
            for process_var, manager_var_name in self.config.stdio_config.env_vars_to_pass.items():
                # print(f"------------------------------------")
                # print(f"Process var: {process_var}")
                # print(f"Manager var name: {manager_var_name}")
                value = self._resolve_env_var(manager_var_name)
                if value is not None:
                    resolved_env[process_var] = value
        # print("####################################")
        # print(f"Resolved env: {resolved_env}")
        return resolved_env
    
    async def _get_auth_headers(self) -> Optional[Dict[str, str]]:
        if self.config.authentication and self.config.authentication.type == "bearer_token":
            if self.config.authentication.token_env_var:
                token = self._resolve_env_var(self.config.authentication.token_env_var)
                if token:
                    return {"Authorization": f"Bearer {token}"}
                else:
                    logger.error(f"Server '{self.config.name}': Bearer token from env var '{self.config.authentication.token_env_var}' not found.")
            else:
                logger.warning(f"Server '{self.config.name}': Bearer token authentication configured but no token_env_var specified.")
        return None

    async def connect(self) -> bool:
        logger.debug(f"Server '{self.config.name}': connect() called. Current status: {self.status}")
        
        try:
            await self._cleanup_resources() # Ensure clean state before new attempt
            self.status = "connecting"

            if self.config.connection_type == "stdio":
                if not self.config.stdio_config:
                    raise ValueError("StdioConfig is missing for stdio connection type.")
                stdio_env = self._get_stdio_env()
                logger.debug(f"Server '{self.config.name}': Initializing StdioTransport with cmd: '{self.config.stdio_config.command}', args: {self.config.stdio_config.args}, env keys: {list(stdio_env.keys())}")
                self._transport = StdioTransport(
                    command=self.config.stdio_config.command,
                    args=self.config.stdio_config.args,
                    env=stdio_env,
                )
                # Perform a quick test to ensure the transport and server can communicate
                async with self._transport.connect_session() as temp_session:
                    await temp_session.list_tools()
                logger.info(f"Server '{self.config.name}' (stdio): Transport initialized and test call successful.")

            elif self.config.connection_type == "url":
                if not self.config.url_config:
                    raise ValueError("UrlConfig is missing for url connection type.")
                auth_headers = await self._get_auth_headers()
                logger.debug(f"Server '{self.config.name}': Initializing FastMCPClient for URL: {self.config.url_config.base_url}")
                
                # Let FastMCP handle its own session management to avoid cancel scope conflicts
                if auth_headers:
                    # Create client with auth headers, let it manage its own session
                    self._http_client = FastMCPClient(base_url=self.config.url_config.base_url, headers=auth_headers)
                else:
                    # No auth headers, simple client creation
                    self._http_client = FastMCPClient(base_url=self.config.url_config.base_url)
                
                await self._http_client.__aenter__() # Enter context for the client itself
                await self._http_client.list_tools() # Verification call
                # Note: _http_client remains active until disconnect. Its __aexit__ will be called in _cleanup_resources.
            else:
                raise ValueError(f"Unsupported connection type: {self.config.connection_type}")

            self.status = "connected"
            self.last_heartbeat = datetime.now(timezone.utc)
            self.reconnect_attempts = 0
            logger.info(f"Server '{self.config.name}': Successfully connected/initialized.")
            return True
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Connection attempt failed. Error: {e}", exc_info=True) # exc_info=True for connection errors
            await self._cleanup_resources()
            self.status = "error"
            return False

    async def _cleanup_resources(self):
        logger.debug(f"Server '{self.config.name}': Cleaning up resources...")
        
        if self._http_client: # For URL client
            try:
                logger.debug(f"Server '{self.config.name}': Closing FastMCPClient (HTTP)...")
                await self._http_client.__aexit__(None, None, None) # Exit context for the client
            except Exception as e:
                logger.debug(f"Server '{self.config.name}': Error exiting http client (this may be expected): {e}")
            self._http_client = None

        # Don't manually close the session if FastMCP is managing it
        # if self._http_client_session and not self._http_client_session.closed: # Our own aiohttp session
        #     try:
        #         logger.debug(f"Server '{self.config.name}': Closing aiohttp.ClientSession...")
        #         await self._http_client_session.close()
        #     except Exception as e:
        #         logger.error(f"Server '{self.config.name}': Error closing aiohttp session: {e}", exc_info=True)
        #     self._http_client_session = None
        self._http_client_session = None  # Just clear the reference

        if self._transport: # For StdioTransport
            try:
                logger.debug(f"Server '{self.config.name}': Closing StdioTransport...")
                # await self._transport.close() # Use transport's close method
            except Exception as e:
                logger.error(f"Server '{self.config.name}': Error closing StdioTransport: {e}", exc_info=True)
            self._transport = None
        
        logger.debug(f"Server '{self.config.name}': Resources cleaned.")

    async def disconnect(self) -> None:
        logger.info(f"Server '{self.config.name}': Disconnecting...")
        await self._cleanup_resources()
        self.status = "disconnected"
        self.reconnect_attempts = 0
        logger.info(f"Server '{self.config.name}': Successfully disconnected.")

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not await self.ensure_connected(): 
            logger.warning(f"Server '{self.config.name}': Cannot list tools, connection failed or not established.")
            return []
        try:
            response = None
            if self.config.connection_type == "stdio":
                if not self._transport:
                    raise ConnectionError(f"Server '{self.config.name}' (stdio): Transport not initialized for list_tools.")
                logger.debug(f"Server '{self.config.name}' (stdio): Acquiring session for list_tools...")
                async with self._transport.connect_session() as session:
                    response = await session.list_tools()
                logger.debug(f"Server '{self.config.name}' (stdio): Session for list_tools released.")
            elif self.config.connection_type == "url" and self._http_client:
                response = await self._http_client.list_tools()
            else:
                raise ConnectionError(f"Server '{self.config.name}': No active client/transport for list_tools after ensure_connected.")
            
            self.last_heartbeat = datetime.now(timezone.utc)
            return response.tools if hasattr(response, 'tools') else response
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Error listing tools: {e}", exc_info=True)
            self.status = "error" # If an operation fails, mark status as error
            return []

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        if not await self.ensure_connected():
            logger.warning(f"Server '{self.config.name}': Cannot call '{tool_name}', connection failed or not established.")
            return None
        try:
            logger.debug(f"Server '{self.config.name}': Calling tool '{tool_name}' with params: {params}")
            result = None
            if self.config.connection_type == "stdio":
                if not self._transport:
                    raise ConnectionError(f"Server '{self.config.name}' (stdio): Transport not initialized for call_tool.")
                logger.debug(f"Server '{self.config.name}' (stdio): Acquiring session for call_tool '{tool_name}'...")
                async with self._transport.connect_session() as session:
                    result = await session.call_tool(name=tool_name, arguments=params)
                logger.debug(f"Server '{self.config.name}' (stdio): Session for call_tool '{tool_name}' released.")
            elif self.config.connection_type == "url" and self._http_client:
                # Assuming FastMCPClient's tool calling mechanism is safe with a long-lived client instance
                if hasattr(self._http_client, 'call_tool'): # Generic call_tool if available
                    result = await self._http_client.call_tool(name=tool_name, arguments=params)
                elif hasattr(self._http_client, 'tools') and hasattr(getattr(self._http_client, 'tools'), tool_name):
                    tool_func = getattr(getattr(self._http_client, 'tools'), tool_name)
                    result = await tool_func.call(**params) if hasattr(tool_func, 'call') else await tool_func(**params)
                else:
                    raise NotImplementedError(f"Tool '{tool_name}' call method not found for URL client '{self.config.name}'.")
            else:
                raise ConnectionError(f"Server '{self.config.name}': No active client/transport for call_tool '{tool_name}' after ensure_connected.")
            
            self.last_heartbeat = datetime.now(timezone.utc)
            return result
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Error calling tool '{tool_name}': {e}", exc_info=True)
            self.status = "error" # If an operation fails, mark status as error
            return None

    async def check_health(self) -> bool:
        # ensure_connected will try to (re)initialize transport/client if needed
        if not await self.ensure_connected():
            logger.warning(f"Server '{self.config.name}': Health check: could not ensure connection/initialization.")
            return False
        
        # If ensure_connected is true, self.status should be "connected".
        # Now perform the actual health check operation.
        logger.debug(f"Server '{self.config.name}': Health check: performing list_tools operation...")
        
        # list_tools will attempt to use the connection. It also sets self.status to "error" on failure.
        await self.list_tools() 
        
        if self.status == "connected":
            logger.debug(f"Server '{self.config.name}': Health check successful (list_tools completed, status is connected).")
            return True
        else: # list_tools call failed and changed the status
            logger.warning(f"Server '{self.config.name}': Health check failed after list_tools. Status: {self.status}")
            return False

    async def ensure_connected(self) -> bool:
        # This method now ensures the transport (for stdio) or http_client (for url) is initialized and responsive.
        if self.status == "connected":
            # Perform a quick lightweight check if already "connected" to ensure it's not stale?
            # For now, assume "connected" status is accurate until next scheduled health check.
            return True
        
        if self.status == "connecting" or self.status == "reconnecting":
            logger.debug(f"Server '{self.config.name}': ensure_connected: Connection/initialization attempt already in progress ({self.status}).")
            return False 

        logger.info(f"Server '{self.config.name}': ensure_connected: Attempting to establish connection/initialize. Retries made: {self.reconnect_attempts}/{self.config.max_reconnect_attempts}")
        
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            if self.reconnect_attempts > 0: # Delay for any retry attempt
                 logger.info(f"Server '{self.config.name}': Delaying for {self.config.reconnect_delay_seconds}s before reconnect attempt {self.reconnect_attempts + 1}...")
                 await asyncio.sleep(self.config.reconnect_delay_seconds)

            self.status = "reconnecting" 
            self.reconnect_attempts += 1
            logger.info(f"Server '{self.config.name}': Starting connection/initialization attempt {self.reconnect_attempts}/{self.config.max_reconnect_attempts}.")

            if await self.connect(): # connect() will set status to "connected" and reset reconnect_attempts on success
                return True
            
            # connect() failed and set status to "error"
            if self.reconnect_attempts >= self.config.max_reconnect_attempts:
                logger.error(f"Server '{self.config.name}': All {self.config.max_reconnect_attempts} connection attempts failed. Final status: {self.status}")
                break 
        
        if self.status != "connected":
            logger.error(f"Server '{self.config.name}': Failed to connect/initialize after {self.reconnect_attempts} attempts. Status: {self.status}")
            self.status = "error" 
        return self.status == "connected"