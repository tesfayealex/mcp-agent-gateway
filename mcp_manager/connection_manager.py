import asyncio
import logging
from typing import List, Dict, Optional

from .config_loader import ServerConfig, load_configs
from .server_handler import MCPClientWrapper

logger = logging.getLogger(__name__)

class MCPConnectionManager:
    def __init__(self, server_configs: Optional[List[ServerConfig]] = None, config_file_path: str = "config.json"):
        if server_configs is None:
            server_configs = load_configs(config_file_path)
        
        self.server_handlers: Dict[str, MCPClientWrapper] = {}
        for conf in server_configs:
            if conf.enabled:
                self.server_handlers[conf.name] = MCPClientWrapper(conf)
                logger.info(f"Initialized MCPClientWrapper for enabled server: {conf.name}")
            else:
                logger.info(f"Skipping disabled server: {conf.name}")
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def connect_all_servers(self):
        if not self.server_handlers:
            logger.warning("No enabled server handlers to connect.")
            return
        logger.info(f"Attempting to connect to {len(self.server_handlers)} enabled MCP server(s)...")
        connect_tasks = [handler.connect() for handler in self.server_handlers.values()]
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        for handler, result in zip(self.server_handlers.values(), results):
            if isinstance(result, Exception):
                logger.error(f"Error connecting to server '{handler.config.name}': {result}")
            elif not result: # connect() returns bool
                logger.warning(f"Failed to connect to server '{handler.config.name}' (connect returned False).")
        logger.info("Finished all initial connection attempts.")

    async def disconnect_all_servers(self):
        if not self.server_handlers:
            logger.info("No server handlers to disconnect.")
            return
        logger.info("Disconnecting from all MCP servers...")
        disconnect_tasks = [handler.disconnect() for handler in self.server_handlers.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True) # Exceptions logged by handler
        logger.info("Finished disconnecting all servers.")

    def get_server_handler(self, name: str) -> Optional[MCPClientWrapper]:
        return self.server_handlers.get(name)

    async def _monitor_servers(self, check_interval_seconds: int):
        logger.info(f"Starting MCP server monitoring task. Check interval: {check_interval_seconds} seconds.")
        while not self._stop_event.is_set():
            try:
                logger.debug("Monitoring cycle: Checking health of all managed servers...")
                health_check_tasks = []
                for handler in self.server_handlers.values():
                    # ensure_connected will attempt to connect if disconnected/error, respecting retries.
                    # check_health will then perform an actual operation if connected.
                    health_check_tasks.append(handler.check_health())
                
                await asyncio.gather(*health_check_tasks, return_exceptions=True) # Errors handled within check_health/ensure_connected

                # Log current statuses after health checks
                for handler_name, handler_obj in self.server_handlers.items():
                    logger.debug(f"Server '{handler_name}' status: {handler_obj.status}, reconnect attempts: {handler_obj.reconnect_attempts}")

                await asyncio.wait_for(self._stop_event.wait(), timeout=check_interval_seconds)
            except asyncio.TimeoutError:
                continue # Loop again for the next check interval
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}", exc_info=True)
                # Avoid rapid looping on unexpected errors
                if not self._stop_event.is_set():
                    await asyncio.sleep(check_interval_seconds) 
        logger.info("MCP server monitoring task stopped.")

    async def start_monitoring(self, check_interval_seconds: int = 60):
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring task already running.")
            return
        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitor_servers(check_interval_seconds))
        logger.info("MCP server monitoring initiated.")

    async def stop_monitoring(self):
        if self._monitoring_task and not self._monitoring_task.done():
            logger.info("Stopping MCP server monitoring task...")
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0) # Give it a moment to stop
            except asyncio.TimeoutError:
                logger.warning("Monitoring task did not stop gracefully within timeout. Attempting to cancel.")
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task # Await cancellation
                except asyncio.CancelledError:
                    logger.info("Monitoring task cancelled successfully.")
            except Exception as e:
                logger.error(f"Error while stopping monitoring task: {e}", exc_info=True)
            self._monitoring_task = None
        else:
            logger.info("Monitoring task not running or already stopped.") 