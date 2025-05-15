import asyncio
import logging
import signal
import os
import argparse
import functools
from typing import Optional

from .config_loader import load_configs
from .connection_manager import MCPConnectionManager

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), # Output to console
        # logging.FileHandler("mcp_manager.log") # Optionally log to a file
    ]
)
logger = logging.getLogger(__name__)

# Global event to signal shutdown, accessible by signal handler callback
shutdown_event_async: Optional[asyncio.Event] = None 
# Prevent multiple shutdown calls
shutdown_in_progress = False

async def perform_graceful_shutdown(manager: MCPConnectionManager):
    global shutdown_in_progress
    if shutdown_in_progress:
        logger.info("Shutdown already in progress, ignoring duplicate signal.")
        return
    shutdown_in_progress = True
    
    logger.info("Graceful shutdown initiated by signal...")
    
    if shutdown_event_async: # Check if event is initialized
        logger.info("Stopping server monitoring...")
        if manager: # ensure manager exists
            await manager.stop_monitoring()
        logger.info("Disconnecting from all servers...")
        if manager:
            await manager.disconnect_all_servers()
        logger.info("Signaling main loop to exit...")
        shutdown_event_async.set()
    else:
        logger.error("Shutdown event not initialized. Cannot proceed with graceful shutdown.")

def signal_handler_callback(signum, manager: MCPConnectionManager):
    logger.info(f"Received signal {signal.Signals(signum).name}. Scheduling graceful shutdown.")
    # Schedule the async shutdown process to be run by the loop
    # Make sure to pass the manager instance if perform_graceful_shutdown needs it.
    asyncio.create_task(perform_graceful_shutdown(manager))

async def main_async(args):
    """Asynchronous main function."""
    global shutdown_event_async # Allow modification of the global event

    logger.info("Starting MCP Agent Gateway...")
    
    # Determine config file path
    # Default to a path relative to this file if not specified by args
    # This makes it easier to run if main.py is not in the project root directly.
    if args.config:
        config_file = args.config
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(base_dir, "config.json")
    
    logger.info(f"Loading server configurations from: {config_file}")
    server_configs = load_configs(config_path=config_file)

    if not server_configs:
        logger.error("No server configurations loaded. MCP Agent Gateway will exit.")
        return

    manager = MCPConnectionManager(server_configs=server_configs)
    shutdown_event_async = asyncio.Event() # Initialize the event here

    loop = asyncio.get_running_loop()
    
    # Use functools.partial to pass the manager to the callback
    handler_with_context = functools.partial(signal_handler_callback, manager=manager)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handler_with_context, sig) # Pass sig to handler

    logger.info("Connecting to all enabled MCP servers...")
    await manager.connect_all_servers()

    logger.info("Starting server monitoring...")
    await manager.start_monitoring(check_interval_seconds=args.interval)

    logger.info("MCP Agent Gateway is running. Press Ctrl+C to stop.")
    
    try:
        await shutdown_event_async.wait() # Keep running until shutdown is signalled
    finally:
        logger.info("Main loop exiting. Cleaning up signal handlers...")
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        # Ensure one final shutdown attempt if not already completed fully, 
        # though perform_graceful_shutdown should handle most of it.
        if not shutdown_in_progress: # If shutdown wasn't triggered by signal for some reason
             logger.info("Main loop cleanup: ensuring shutdown tasks run.")
             await perform_graceful_shutdown(manager)

    logger.info("MCP Agent Gateway has shut down.")

def main():
    parser = argparse.ArgumentParser(description="MCP Agent Gateway to manage connections to multiple MCP servers.")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to the JSON configuration file for MCP servers. Defaults to 'config.json' in the project root."
    )
    parser.add_argument(
        "-i", "--interval", 
        type=int, 
        default=60, 
        help="Monitoring interval in seconds for health checks. Default is 60."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main() 