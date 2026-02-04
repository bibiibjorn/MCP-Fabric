"""
Test script to check if DAX Execute Queries API is enabled in your tenant.
Run this to verify the tenant setting before implementing the MCP tools.
"""

import asyncio
import requests
from helpers.utils.authentication import get_azure_credentials

# Power BI API scope (different from Fabric API)
POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"


def get_powerbi_token():
    """Get access token for Power BI API."""
    credential = get_azure_credentials("test-client", {})
    token = credential.get_token(POWERBI_SCOPE)
    return token.token


def test_execute_queries(workspace_id: str, dataset_id: str):
    """
    Test the Execute Queries API endpoint.

    Args:
        workspace_id: The workspace/group ID (GUID)
        dataset_id: The dataset/semantic model ID (GUID)
    """
    token = get_powerbi_token()

    url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets/{dataset_id}/executeQueries"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Simple test query - just get row count from first table
    body = {
        "queries": [
            {
                "query": "EVALUATE ROW(\"Test\", 1)"
            }
        ],
        "serializerSettings": {
            "includeNulls": True
        }
    }

    print(f"Testing Execute Queries API...")
    print(f"Workspace: {workspace_id}")
    print(f"Dataset: {dataset_id}")
    print(f"URL: {url}")
    print()

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        print("✓ SUCCESS! The tenant setting is ENABLED.")
        print(f"Response: {response.json()}")
        return True
    elif response.status_code == 403:
        error = response.json()
        print("✗ FORBIDDEN - Tenant setting may be disabled or you lack permissions.")
        print(f"Error: {error}")
        return False
    elif response.status_code == 400:
        error = response.json()
        if "Execute Queries API" in str(error):
            print("✗ DISABLED - The 'Dataset Execute Queries REST API' tenant setting is NOT enabled.")
        else:
            print(f"✗ BAD REQUEST - Check your query syntax or dataset ID.")
        print(f"Error: {error}")
        return False
    else:
        print(f"✗ Unexpected status: {response.status_code}")
        print(f"Response: {response.text}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Power BI Execute Queries API - Tenant Setting Test")
    print("=" * 60)
    print()

    # You need to provide these values
    # Get them from list_workspaces and list_semantic_models tools
    WORKSPACE_ID = input("Enter Workspace ID (GUID): ").strip()
    DATASET_ID = input("Enter Dataset/Semantic Model ID (GUID): ").strip()

    if not WORKSPACE_ID or not DATASET_ID:
        print("Error: Both Workspace ID and Dataset ID are required.")
        exit(1)

    test_execute_queries(WORKSPACE_ID, DATASET_ID)
