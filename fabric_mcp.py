from tools import *
from helpers.logging_config import get_logger
from helpers.utils.context import mcp, __ctx_cache
import uvicorn
import argparse
import logging
import os
import sys

logger = get_logger(__name__)
logger.level = logging.INFO


def ensure_authenticated():
    """
    Ensure Azure authentication is set up before starting the MCP server.
    If no cached credentials exist, opens browser for interactive login.
    """
    from azure.identity import (
        InteractiveBrowserCredential,
        TokenCachePersistenceOptions,
        AuthenticationRecord,
    )

    AUTH_RECORD_PATH = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        ".fabric_mcp_python",
        "auth_record.json"
    )
    FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"

    cache_options = TokenCachePersistenceOptions(
        name="fabric_mcp_python",
        allow_unencrypted_storage=True,
    )

    # Check if we have cached credentials
    auth_record = None
    if os.path.exists(AUTH_RECORD_PATH):
        try:
            with open(AUTH_RECORD_PATH, "r") as f:
                auth_record = AuthenticationRecord.deserialize(f.read())
            logger.info(f"Found cached credentials for: {auth_record.username}")
        except Exception as e:
            logger.warning(f"Failed to load cached credentials: {e}")

    # If no cached credentials, need to authenticate
    if auth_record is None:
        logger.info("No cached credentials found. Starting interactive authentication...")
        logger.info("A browser window will open for you to sign in to Azure.")

        cred = InteractiveBrowserCredential(cache_persistence_options=cache_options)

        try:
            # This opens browser for authentication
            auth_record = cred.authenticate(scopes=[FABRIC_SCOPE])

            # Save for future use
            os.makedirs(os.path.dirname(AUTH_RECORD_PATH), exist_ok=True)
            with open(AUTH_RECORD_PATH, "w") as f:
                f.write(auth_record.serialize())

            logger.info(f"Authentication successful! Logged in as: {auth_record.username}")
            logger.info(f"Credentials cached at: {AUTH_RECORD_PATH}")

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("Please ensure you have access to Microsoft Fabric.")
            sys.exit(1)
    else:
        # Verify cached credentials still work
        try:
            cred = InteractiveBrowserCredential(
                cache_persistence_options=cache_options,
                authentication_record=auth_record,
            )
            # Try to get a token silently
            token = cred.get_token(FABRIC_SCOPE)
            logger.info("Cached credentials verified successfully.")
        except Exception as e:
            logger.warning(f"Cached credentials expired or invalid: {e}")
            logger.info("Re-authenticating...")

            # Remove old record and re-authenticate
            try:
                os.remove(AUTH_RECORD_PATH)
            except:
                pass

            cred = InteractiveBrowserCredential(cache_persistence_options=cache_options)
            try:
                auth_record = cred.authenticate(scopes=[FABRIC_SCOPE])
                os.makedirs(os.path.dirname(AUTH_RECORD_PATH), exist_ok=True)
                with open(AUTH_RECORD_PATH, "w") as f:
                    f.write(auth_record.serialize())
                logger.info(f"Re-authentication successful! Logged in as: {auth_record.username}")
            except Exception as e:
                logger.error(f"Re-authentication failed: {e}")
                sys.exit(1)


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
        ensure_authenticated()

    if args.port:
        # Start the server with Streamable HTTP transport
        uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=args.port)
    else:
        # Default to STDIO mode for Claude Desktop
        mcp.run(transport="stdio")
