from azure.identity import (
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)
from cachetools import TTLCache
from datetime import datetime, timedelta
import json
import os
import sys
import logging

logger = logging.getLogger(__name__)

# How often to require re-authentication (in hours)
REAUTH_INTERVAL_HOURS = 24

# Path to store authentication record for persistence across restarts
AUTH_RECORD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    ".fabric_mcp_python",
    "auth_record.json"
)

# Path to store last login timestamp
LAST_LOGIN_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    ".fabric_mcp_python",
    "last_login.json"
)

FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
SQL_SCOPE = "https://database.windows.net/.default"
POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"

# Singleton credential instance for both REST API and SQL/TDS connections
_shared_credential = None
_shared_auth_record = None

# Cache options used throughout - uses DPAPI encryption on Windows
_cache_options = TokenCachePersistenceOptions(
    name="fabric_mcp_python",
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


def _save_last_login():
    """Save the current timestamp as last login time."""
    os.makedirs(os.path.dirname(LAST_LOGIN_PATH), exist_ok=True)
    with open(LAST_LOGIN_PATH, "w") as f:
        json.dump({"last_login": datetime.now().isoformat()}, f)


def _is_login_expired() -> bool:
    """Check if the last login was more than REAUTH_INTERVAL_HOURS ago."""
    try:
        if os.path.exists(LAST_LOGIN_PATH):
            with open(LAST_LOGIN_PATH, "r") as f:
                data = json.load(f)
                last_login = datetime.fromisoformat(data["last_login"])
                age = datetime.now() - last_login
                if age > timedelta(hours=REAUTH_INTERVAL_HOURS):
                    logger.info(f"Last login was {age.total_seconds() / 3600:.1f} hours ago - re-authentication required")
                    return True
                else:
                    logger.info(f"Last login was {age.total_seconds() / 3600:.1f} hours ago - still valid")
                    return False
    except Exception as e:
        logger.warning(f"Could not read last login time: {e}")
    # No record means we need to login
    return True


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
        _save_last_login()  # Record when this login happened
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

    # Check if daily re-authentication is required
    login_expired = _is_login_expired()

    # Try to load existing authentication record
    auth_record = _load_auth_record()

    if auth_record is None or login_expired:
        # No cached credentials OR daily login required - perform interactive auth
        if login_expired and auth_record is not None:
            logger.info("Daily re-authentication required.")
        else:
            logger.info("No cached credentials found.")
        _shared_credential, _shared_auth_record = _perform_interactive_auth()
        return _shared_credential

    # Have cached credentials that are still within daily window - try to use them
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
