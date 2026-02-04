from tools import *
from helpers.logging_config import get_logger
from helpers.utils.context import mcp, __ctx_cache
from helpers.utils.authentication import ensure_authenticated, AuthenticationError
import uvicorn
import argparse
import logging
import sys

logger = get_logger(__name__)
logger.level = logging.INFO


@mcp.tool()
async def clear_context() -> str:
    """Clear the current session context.

    Returns:
        A string confirming the context has been cleared.
    """
    __ctx_cache.clear()
    return "Context cleared."


if __name__ == "__main__":
    # Initialize and run the server
    logger.info("Starting MCP server...")
    parser = argparse.ArgumentParser(description="Run MCP server")
    parser.add_argument("--port", type=int, default=None, help="Localhost port to listen on (HTTP mode)")
    parser.add_argument("--stdio", action="store_true", help="Run in STDIO mode for Claude Desktop")
    parser.add_argument("--skip-auth", action="store_true", help="Skip authentication check at startup")
    args = parser.parse_args()

    # Ensure authentication before starting
    if not args.skip_auth:
        try:
            ensure_authenticated()
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("Please ensure you have access to Microsoft Fabric.")
            sys.exit(1)

    if args.port:
        # Start the server with Streamable HTTP transport
        uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=args.port)
    else:
        # Default to STDIO mode for Claude Desktop
        mcp.run(transport="stdio")
