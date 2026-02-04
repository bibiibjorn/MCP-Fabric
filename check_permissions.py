"""
Check your permissions on a Fabric workspace and test notebook access.
Run this to diagnose permission issues with get_notebook_content.
"""

import asyncio
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.clients.fabric_client import FabricApiClient
from helpers.utils.authentication import get_azure_credentials, AUTH_RECORD_PATH
from cachetools import TTLCache


async def check_permissions(workspace_name: str):
    """Check permissions on a workspace and test notebook access."""

    cache = TTLCache(maxsize=100, ttl=3600)

    print(f"\n{'='*60}")
    print("FABRIC PERMISSION CHECK")
    print(f"{'='*60}")
    print(f"\nAuth record path: {AUTH_RECORD_PATH}")
    print(f"Auth record exists: {os.path.exists(AUTH_RECORD_PATH)}")

    try:
        creds = get_azure_credentials("test_client", cache)
        print("[OK] Credentials loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load credentials: {e}")
        return

    client = FabricApiClient(creds)

    # Test 1: Resolve workspace
    print(f"\n--- Testing workspace access: {workspace_name} ---")
    try:
        workspace_id = await client.resolve_workspace(workspace_name)
        print(f"[OK] Workspace resolved: {workspace_id}")
    except Exception as e:
        print(f"[FAIL] Failed to resolve workspace: {e}")
        return

    # Test 2: Get workspace details (shows your role)
    try:
        # List all workspaces and find this one to see role info
        workspaces = await client.get_workspaces()
        for ws in workspaces:
            if ws.get("id") == workspace_id:
                print(f"\n[INFO] Workspace Details:")
                print(f"   Name: {ws.get('displayName')}")
                print(f"   ID: {ws.get('id')}")
                print(f"   Type: {ws.get('type', 'N/A')}")
                print(f"   Capacity ID: {ws.get('capacityId', 'N/A')}")
                # Note: Role info might not be available in all API responses
                break
    except Exception as e:
        print(f"[WARN]  Could not get workspace details: {e}")

    # Test 3: List notebooks
    print(f"\n--- Testing notebook list access ---")
    try:
        notebooks = await client.get_notebooks(workspace_id)
        print(f"[OK] Found {len(notebooks)} notebooks")

        if notebooks:
            # Show first 5 notebooks
            for nb in notebooks[:5]:
                print(f"   - {nb.get('displayName')} ({nb.get('id')})")
            if len(notebooks) > 5:
                print(f"   ... and {len(notebooks) - 5} more")
    except Exception as e:
        print(f"[FAIL] Failed to list notebooks: {e}")
        return

    # Test 4: Try to get a notebook definition (this requires write permission)
    if notebooks:
        test_notebook = notebooks[0]
        nb_id = test_notebook.get('id')
        nb_name = test_notebook.get('displayName')

        # Also test the get_notebook_content tool
        print(f"\n--- Testing get_notebook_content tool ---")
        try:
            from helpers.clients.notebook_client import NotebookClient
            import base64

            notebook_client = NotebookClient(client)
            definition = await notebook_client.get_notebook_definition(workspace_id, nb_id)

            if definition and isinstance(definition, dict):
                parts = definition.get("definition", {}).get("parts", [])
                for part in parts:
                    path = part.get("path", "")
                    if path == "notebook-content.py" or path.endswith(".py"):
                        payload = part.get("payload", "")
                        if payload:
                            decoded = base64.b64decode(payload).decode("utf-8")
                            print(f"[OK] Successfully decoded notebook content ({len(decoded)} chars)")
                            print(f"   First 200 chars:\n   {decoded[:200]}...")
                            break
                    elif path.endswith(".ipynb"):
                        payload = part.get("payload", "")
                        if payload:
                            decoded = base64.b64decode(payload).decode("utf-8")
                            print(f"[OK] Successfully decoded notebook content (.ipynb, {len(decoded)} chars)")
                            break
        except Exception as e:
            print(f"[FAIL] get_notebook_content test failed: {e}")

        print(f"\n--- Testing getDefinition on '{nb_name}' ---")
        print("(This requires read+write permissions)")

        try:
            definition = await client.get_notebook_definition(workspace_id, nb_id)

            if definition:
                print(f"[OK] SUCCESS! Got notebook definition")
                print(f"   You have READ+WRITE permissions on this workspace")

                # Debug: show full response structure
                print(f"\n   Full response keys: {list(definition.keys())}")
                import json
                print(f"   Full response (truncated):")
                response_str = json.dumps(definition, indent=2)
                # Show first 2000 chars
                if len(response_str) > 2000:
                    print(f"   {response_str[:2000]}...")
                else:
                    print(f"   {response_str}")

                # Show structure
                parts = definition.get("definition", {}).get("parts", [])
                print(f"\n   Definition has {len(parts)} parts:")
                for part in parts:
                    path = part.get("path", "unknown")
                    payload_size = len(part.get("payload", ""))
                    print(f"     - {path} ({payload_size} chars base64)")
            else:
                print(f"[FAIL] getDefinition returned None/empty")
                print(f"   This typically means insufficient permissions")
        except Exception as e:
            error_str = str(e)
            print(f"[FAIL] getDefinition failed: {e}")

            if "403" in error_str or "Forbidden" in error_str:
                print("\n[DIAGNOSIS] DIAGNOSIS: Permission denied (403 Forbidden)")
                print("   The getDefinition API requires BOTH read AND write permissions.")
                print("   Your current access level is READ-ONLY.")
                print("\n   Solutions:")
                print("   1. Ask a workspace admin to upgrade you to Contributor or higher")
                print("   2. Or use a service principal with Notebook.ReadWrite.All scope")
            elif "401" in error_str or "Unauthorized" in error_str:
                print("\n[DIAGNOSIS] DIAGNOSIS: Authentication issue")
                print("   Try re-running: python authenticate.py")
            else:
                print("\n[DIAGNOSIS] DIAGNOSIS: Unknown error")
                print(f"   Full error: {error_str}")

    print(f"\n{'='*60}")
    print("CHECK COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "FTX-DEV-MGT"
    print(f"Checking permissions on workspace: {workspace}")
    asyncio.run(check_permissions(workspace))
