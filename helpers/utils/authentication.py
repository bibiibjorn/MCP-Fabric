from azure.identity import (
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)
from cachetools import TTLCache
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Path to store authentication record for persistence across restarts
AUTH_RECORD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    ".fabric_mcp_python",
    "auth_record.json"
)

FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
SQL_SCOPE = "https://database.windows.net/.default"
POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"

# Singleton credential instance for both REST API and SQL/TDS connections
_shared_credential = None
_shared_auth_record = None

# Cache options used throughout
_cache_options = TokenCachePersistenceOptions(
    name="fabric_mcp_python",
    allow_unencrypted_storage=True,
)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
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


def _save_auth_record(auth_record: AuthenticationRecord):
    """Save authentication record for future use."""
    os.makedirs(os.path.dirname(AUTH_RECORD_PATH), exist_ok=True)
    with open(AUTH_RECORD_PATH, "w") as f:
        f.write(auth_record.serialize())


def _perform_interactive_auth() -> tuple:
    """
    Perform interactive browser authentication.

    Returns:
        Tuple of (credential, auth_record)
    """
    logger.info("Starting interactive authentication - browser will open...")

    cred = InteractiveBrowserCredential(cache_persistence_options=_cache_options)

    try:
        # This opens browser for authentication
        auth_record = cred.authenticate(scopes=[FABRIC_SCOPE])
        _save_auth_record(auth_record)
        logger.info(f"Authentication successful! Logged in as: {auth_record.username}")
        return cred, auth_record
    except Exception as e:
        raise AuthenticationError(f"Interactive authentication failed: {e}")


def get_shared_credential():
    """
    Get the shared credential instance for both REST API and SQL/TDS connections.

    This function automatically handles authentication:
    - If cached credentials exist and are valid, uses them silently
    - If no credentials or expired, opens browser for interactive login
    - Credentials are cached for future use

    This is a singleton that ensures we use the same cached auth for:
    - Fabric REST API (api.fabric.microsoft.com)
    - SQL Analytics endpoints (database.windows.net)

    Returns:
        InteractiveBrowserCredential: The shared credential instance.

    Raises:
        AuthenticationError: If authentication fails.
    """
    global _shared_credential, _shared_auth_record

    if _shared_credential is not None:
        return _shared_credential

    # Try to load existing authentication record
    auth_record = _load_auth_record()

    if auth_record is None:
        # No cached credentials - perform interactive auth
        logger.info("No cached credentials found.")
        _shared_credential, _shared_auth_record = _perform_interactive_auth()
        return _shared_credential

    # Have cached credentials - try to use them
    logger.info(f"Found cached credentials for: {auth_record.username}")

    cred = InteractiveBrowserCredential(
        cache_persistence_options=_cache_options,
        authentication_record=auth_record,
    )

    # Verify credentials still work by getting a token
    try:
        cred.get_token(FABRIC_SCOPE)
        logger.info("Cached credentials verified successfully.")
        _shared_credential = cred
        _shared_auth_record = auth_record
        return _shared_credential
    except Exception as e:
        # Credentials expired or invalid - re-authenticate
        logger.warning(f"Cached credentials expired or invalid: {e}")
        logger.info("Re-authenticating...")

        # Remove old record
        try:
            os.remove(AUTH_RECORD_PATH)
        except Exception:
            pass

        _shared_credential, _shared_auth_record = _perform_interactive_auth()
        return _shared_credential


def get_azure_credentials(client_id: str, cache: TTLCache):
    """
    Get Azure credentials using cached authentication.

    DEPRECATED: Use get_shared_credential() instead for unified auth.

    Uses persistent token caching so authentication only needs to happen once.
    Tokens are automatically refreshed using the cached refresh token.
    """
    if f"{client_id}_creds" in cache:
        return cache[f"{client_id}_creds"]

    # Use the shared credential
    creds = get_shared_credential()

    cache[f"{client_id}_creds"] = creds
    return cache[f"{client_id}_creds"]


def ensure_authenticated():
    """
    Ensure Azure authentication is set up.

    This is called at MCP server startup to ensure credentials are ready.
    If no cached credentials exist, opens browser for interactive login.

    This is the single entry point for authentication - all other functions
    use get_shared_credential() which delegates to this logic.
    """
    # Simply call get_shared_credential - it handles everything
    get_shared_credential()
