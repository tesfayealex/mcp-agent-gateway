import asyncio
import logging
import signal
import os
import argparse

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

async def main_async(args):
    """Asynchronous main function."""
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
    server_configs = load_configs(config_file_path=config_file)

    if not server_configs:
        logger.error("No server configurations loaded. MCP Agent Gateway will exit.")
        return

    manager = MCPConnectionManager(server_configs=server_configs) # Pass configs directly

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event_sync = asyncio.Event() # Used to signal main to exit from signal handler

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signal.Signals(signum).name}. Initiating graceful shutdown...")
        # Schedule the async shutdown process to be run by the loop
        asyncio.create_task(shutdown(manager, stop_event_sync))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Connecting to all enabled MCP servers...")
    await manager.connect_all_servers()

    logger.info("Starting server monitoring...")
    await manager.start_monitoring(check_interval_seconds=args.interval)

    logger.info("MCP Agent Gateway is running. Press Ctrl+C to stop.")
    await stop_event_sync.wait() # Keep running until shutdown is signalled

    logger.info("MCP Agent Gateway has shut down.")

async def shutdown(manager: MCPConnectionManager, stop_event_sync: asyncio.Event):
    """Handles the graceful shutdown sequence."""
    logger.info("Stopping server monitoring...")
    await manager.stop_monitoring()
    logger.info("Disconnecting from all servers...")
    await manager.disconnect_all_servers()
    logger.info("Setting stop event for main loop...")
    stop_event_sync.set() # Signal the main loop to exit

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