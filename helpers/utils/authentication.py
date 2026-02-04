from azure.identity import (
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)
from cachetools import TTLCache
import os
import sys

# Path to store authentication record for persistence across restarts
AUTH_RECORD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    ".fabric_mcp_python",
    "auth_record.json"
)

FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"


class AuthenticationError(Exception):
    """Raised when authentication is not configured."""
    pass


def _load_auth_record():
    """Load saved authentication record if it exists."""
    try:
        if os.path.exists(AUTH_RECORD_PATH):
            with open(AUTH_RECORD_PATH, "r") as f:
                return AuthenticationRecord.deserialize(f.read())
    except Exception:
        pass
    return None


def get_azure_credentials(client_id: str, cache: TTLCache):
    """
    Get Azure credentials using cached authentication.

    IMPORTANT: You must run 'python authenticate.py' first to set up credentials.
    The MCP server cannot open browser windows for authentication.

    Uses persistent token caching so authentication only needs to happen once.
    Tokens are automatically refreshed using the cached refresh token.
    """
    if f"{client_id}_creds" in cache:
        return cache[f"{client_id}_creds"]

    # Try to load existing authentication record
    auth_record = _load_auth_record()

    if auth_record is None:
        error_msg = (
            "\n" + "=" * 60 + "\n"
            "AUTHENTICATION REQUIRED\n"
            "=" * 60 + "\n"
            "The Fabric MCP server needs Azure authentication.\n\n"
            "Please run the authentication script first:\n\n"
            "  cd \"C:\\Users\\bjorn.braet\\powerbi-mcp-servers\\MCP-Fabric-Python Based\"\n"
            "  .venv\\Scripts\\python.exe authenticate.py\n\n"
            "This only needs to be done once. After authenticating,\n"
            "restart Claude Desktop and the MCP server will work.\n"
            "=" * 60 + "\n"
        )
        print(error_msg, file=sys.stderr)
        raise AuthenticationError("Authentication not configured. Run 'python authenticate.py' first.")

    # Enable persistent token caching - tokens survive restarts
    cache_options = TokenCachePersistenceOptions(
        name="fabric_mcp_python",  # Unique cache name for this app
        allow_unencrypted_storage=True,  # Required on some systems without encryption
    )

    # Use InteractiveBrowserCredential with the saved auth record
    # This allows silent token refresh without opening a browser
    creds = InteractiveBrowserCredential(
        cache_persistence_options=cache_options,
        authentication_record=auth_record,
    )

    cache[f"{client_id}_creds"] = creds
    return cache[f"{client_id}_creds"]
