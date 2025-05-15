import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from fastmcp.client import StdioTransport, Client as FastMCPClient, Session as FastMCPSession # Assuming Session for stdio
import aiohttp # For URL client session management if needed

from .config_loader import ServerConfig, StdioConfig, UrlConfig, AuthConfig

logger = logging.getLogger(__name__)

class MCPClientWrapper:
    def __init__(self, server_config: ServerConfig):
        self.config: ServerConfig = server_config
        self.status: str = "disconnected"  # disconnected, connecting, connected, error, reconnecting
        self._transport: Optional[StdioTransport] = None
        self._stdio_session: Optional[FastMCPSession] = None # For stdio connections
        self._http_client: Optional[FastMCPClient] = None # For URL connections
        self._http_client_session: Optional[aiohttp.ClientSession] = None # For managing underlying aiohttp session for URL client
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
            for process_var, manager_var_name in self.config.stdio_config.env_vars_to_pass.items():
                value = self._resolve_env_var(manager_var_name)
                if value is not None: # Only pass var if it resolved
                    resolved_env[process_var] = value
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
        # This method attempts a single connection. Retries are handled by ensure_connected.
        # self.status is typically "connecting" or "reconnecting" when this is called by ensure_connected.

        try:
            # Ensure resources are clean before a new attempt
            await self._cleanup_resources() # Important for retry scenarios
            
            self.status = "connecting" # Set status for this attempt

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
                self._stdio_session = await self._transport.connect_session().__aenter__()
                await self._stdio_session.list_tools() # Verification call

            elif self.config.connection_type == "url":
                if not self.config.url_config:
                    raise ValueError("UrlConfig is missing for url connection type.")
                auth_headers = await self._get_auth_headers()
                logger.debug(f"Server '{self.config.name}': Initializing FastMCPClient for URL: {self.config.url_config.base_url}")
                if auth_headers:
                    self._http_client_session = aiohttp.ClientSession(headers=auth_headers)
                    self._http_client = FastMCPClient(base_url=self.config.url_config.base_url, session=self._http_client_session)
                else:
                    self._http_client = FastMCPClient(base_url=self.config.url_config.base_url)
                await self._http_client.__aenter__()
                await self._http_client.list_tools() # Verification call
            else:
                raise ValueError(f"Unsupported connection type: {self.config.connection_type}")

            self.status = "connected"
            self.last_heartbeat = datetime.now(timezone.utc)
            self.reconnect_attempts = 0 # Reset on any successful connect
            logger.info(f"Server '{self.config.name}': Successfully connected.")
            return True
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Connection attempt failed. Error: {e}", exc_info=False)
            await self._cleanup_resources() # Ensure cleanup on failure
            self.status = "error" # Mark as error for this attempt
            return False

    async def _cleanup_resources(self):
        logger.debug(f"Server '{self.config.name}': Cleaning up resources...")
        if self._http_client:
            try: await self._http_client.__aexit__(None, None, None)
            except Exception as e: logger.error(f"Server '{self.config.name}': Error exiting http client: {e}")
            self._http_client = None
        if self._http_client_session and not self._http_client_session.closed:
            try: await self._http_client_session.close()
            except Exception as e: logger.error(f"Server '{self.config.name}': Error closing aiohttp session: {e}")
            self._http_client_session = None
        if self._stdio_session and self._transport: # For stdio, transport manages session lifecyle via context
            try: await self._transport.connect_session().__aexit__(None, None, None)
            except Exception as e: logger.error(f"Server '{self.config.name}': Error exiting stdio session via transport: {e}")
        self._stdio_session = None
        # StdioTransport itself doesn't have an explicit close if subprocess is managed by Popen and terminated.
        # If StdioTransport.close() is available and needed for subprocesses, it should be called.
        # For now, relying on __aexit__ of connect_session() and process termination for cleanup.
        self._transport = None 

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
            if self.config.connection_type == "stdio" and self._stdio_session:
                response = await self._stdio_session.list_tools()
            elif self.config.connection_type == "url" and self._http_client:
                response = await self._http_client.list_tools()
            else:
                raise ConnectionError("No active session/client for list_tools after ensure_connected.")
            self.last_heartbeat = datetime.now(timezone.utc)
            return response.tools if hasattr(response, 'tools') else response
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Error listing tools: {e}", exc_info=True)
            self.status = "error"
            return []

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        if not await self.ensure_connected():
            logger.warning(f"Server '{self.config.name}': Cannot call '{tool_name}', connection failed or not established.")
            return None
        try:
            logger.debug(f"Server '{self.config.name}': Calling tool '{tool_name}' with params: {params}")
            result = None
            if self.config.connection_type == "stdio" and self._stdio_session:
                result = await self._stdio_session.call_tool(name=tool_name, arguments=params)
            elif self.config.connection_type == "url" and self._http_client:
                if hasattr(self._http_client, 'call_tool'):
                    result = await self._http_client.call_tool(name=tool_name, arguments=params)
                elif hasattr(self._http_client, 'tools') and hasattr(getattr(self._http_client, 'tools'), tool_name):
                    tool_func = getattr(getattr(self._http_client, 'tools'), tool_name)
                    result = await tool_func.call(**params) if hasattr(tool_func, 'call') else await tool_func(**params)
                else:
                    raise NotImplementedError(f"Tool '{tool_name}' call method not found for URL client '{self.config.name}'.")
            else:
                raise ConnectionError(f"No active session/client for call_tool '{tool_name}' after ensure_connected.")
            self.last_heartbeat = datetime.now(timezone.utc)
            return result
        except Exception as e:
            logger.error(f"Server '{self.config.name}': Error calling tool '{tool_name}': {e}", exc_info=True)
            self.status = "error"
            return None

    async def check_health(self) -> bool:
        if not await self.ensure_connected():
            logger.warning(f"Server '{self.config.name}': Health check: could not ensure connection.")
            return False
        logger.debug(f"Server '{self.config.name}': Health check: performing list_tools operation...")
        if await self.list_tools() is not None: # list_tools returns [] on error, so check for not None
            if self.status == "connected": # list_tools might change status on its own error
                logger.debug(f"Server '{self.config.name}': Health check successful.")
                return True
        logger.warning(f"Server '{self.config.name}': Health check failed (list_tools unsuccessful or status changed). Status: {self.status}")
        return False

    async def ensure_connected(self) -> bool:
        if self.status == "connected":
            return True
        if self.status == "connecting" or self.status == "reconnecting":
            logger.debug(f"Server '{self.config.name}': ensure_connected: Connection attempt already in progress ({self.status}).")
            return False 

        logger.info(f"Server '{self.config.name}': ensure_connected: Attempting to establish connection. Retries made so far: {self.reconnect_attempts}")
        
        current_attempt = 0 # Tracks attempts within this call to ensure_connected
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            current_attempt +=1
            if self.reconnect_attempts > 0 or current_attempt > 1: # Don't delay for the very first actual attempt in the lifecycle of this wrapper, unless it's a retry.
                 logger.info(f"Server '{self.config.name}': Delaying for {self.config.reconnect_delay_seconds}s before reconnect attempt {self.reconnect_attempts + 1}...")
                 await asyncio.sleep(self.config.reconnect_delay_seconds)

            self.status = "reconnecting" # Explicitly set before connect call
            self.reconnect_attempts += 1
            logger.info(f"Server '{self.config.name}': Starting connection attempt {self.reconnect_attempts}/{self.config.max_reconnect_attempts}.")

            if await self.connect(): # connect() now resets its own attempts on success and sets status
                return True
            
            # connect() failed and set status to "error"
            if self.reconnect_attempts >= self.config.max_reconnect_attempts:
                logger.error(f"Server '{self.config.name}': All {self.config.max_reconnect_attempts} reconnection attempts failed. Final status: {self.status}")
                break # Exit while loop
            # else: continue loop for next attempt
        
        if self.status != "connected":
            logger.error(f"Server '{self.config.name}': Failed to connect after {self.reconnect_attempts} attempts. Giving up for now. Status: {self.status}")
            self.status = "error" # Final confirmation of error status
        return False