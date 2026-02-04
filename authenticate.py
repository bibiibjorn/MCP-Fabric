#!/usr/bin/env python
"""
Fabric MCP Authentication Script

Run this script manually to authenticate with Azure/Fabric.
The credentials will be cached and used by the MCP server.

Usage:
    python authenticate.py

After running this once, the MCP server will use the cached credentials
and won't need browser popups.
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.identity import InteractiveBrowserCredential, TokenCachePersistenceOptions, AuthenticationRecord

# Same paths as the MCP server uses
AUTH_RECORD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    ".fabric_mcp_python",
    "auth_record.json"
)

FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"


def authenticate():
    """Authenticate with Azure and cache credentials for the MCP server."""

    print("=" * 60)
    print("Fabric MCP Authentication")
    print("=" * 60)
    print()

    # Check for existing auth record
    if os.path.exists(AUTH_RECORD_PATH):
        print(f"Existing auth record found at: {AUTH_RECORD_PATH}")
        response = input("Do you want to re-authenticate? (y/N): ").strip().lower()
        if response != 'y':
            print("\nTesting existing credentials...")
            try:
                with open(AUTH_RECORD_PATH, "r") as f:
                    auth_record = AuthenticationRecord.deserialize(f.read())

                cache_options = TokenCachePersistenceOptions(
                    name="fabric_mcp_python",
                    allow_unencrypted_storage=True,
                )

                cred = InteractiveBrowserCredential(
                    cache_persistence_options=cache_options,
                    authentication_record=auth_record,
                )

                # Try to get a token silently
                token = cred.get_token(FABRIC_SCOPE)
                print(f"\n✓ Credentials are valid!")
                print(f"  User: {auth_record.username}")
                print(f"  Tenant: {auth_record.tenant_id}")
                print(f"  Token expires: {token.expires_on}")
                print("\nThe MCP server should work without any issues.")
                return
            except Exception as e:
                print(f"\n✗ Existing credentials are invalid or expired: {e}")
                print("  Re-authenticating...")

    print("\nA browser window will open for you to sign in to Azure.")
    print("This is a one-time setup - credentials will be cached.\n")

    # Create credential with persistent cache
    cache_options = TokenCachePersistenceOptions(
        name="fabric_mcp_python",
        allow_unencrypted_storage=True,
    )

    cred = InteractiveBrowserCredential(
        cache_persistence_options=cache_options,
    )

    try:
        # Authenticate and get the record
        print("Opening browser for authentication...")
        auth_record = cred.authenticate(scopes=[FABRIC_SCOPE])

        # Save the auth record
        os.makedirs(os.path.dirname(AUTH_RECORD_PATH), exist_ok=True)
        with open(AUTH_RECORD_PATH, "w") as f:
            f.write(auth_record.serialize())

        # Verify it works
        token = cred.get_token(FABRIC_SCOPE)

        print()
        print("=" * 60)
        print("✓ Authentication successful!")
        print("=" * 60)
        print(f"  User: {auth_record.username}")
        print(f"  Tenant: {auth_record.tenant_id}")
        print(f"  Auth record saved to: {AUTH_RECORD_PATH}")
        print()
        print("The MCP server will now use these cached credentials.")
        print("You can restart Claude Desktop and the server should work.")
        print()
        print("Note: You may need to re-run this script if:")
        print("  - Your refresh token expires (typically 90 days)")
        print("  - You change your password")
        print("  - Your organization revokes access")

    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
        print("\nPlease try again or check your Azure permissions.")
        sys.exit(1)


if __name__ == "__main__":
    authenticate()
