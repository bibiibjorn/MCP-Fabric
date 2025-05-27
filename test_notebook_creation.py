#!/usr/bin/env python3
"""
Test script to validate the notebook creation fixes
"""
import asyncio
import sys
import json
from helpers.clients.fabric_client import FabricApiClient
from helpers.clients.notebook_client import NotebookClient
from helpers.utils.authentication import get_azure_credentials
from helpers.logging_config import get_logger

logger = get_logger(__name__)

async def test_notebook_creation():
    """Test notebook creation with improved error handling"""
    try:
        # Initialize clients
        credentials = get_azure_credentials("test-client-id", {})
        fabric_client = FabricApiClient(credentials)
        notebook_client = NotebookClient(fabric_client)
        
        # Test workspace - using "My workspace" 
        workspace_id = "645f0acc-fd1e-42fe-ae6e-e919b6c63322"
        notebook_name = "Test Debug Notebook"
        
        # Create a simple notebook content
        notebook_json = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('Hello, Fabric!')\n"],
                    "execution_count": None,
                    "outputs": [],
                    "metadata": {},
                }
            ],
            "metadata": {"language_info": {"name": "python"}},
        }
        notebook_content = json.dumps(notebook_json)
        
        print(f"Testing notebook creation in workspace: {workspace_id}")
        print(f"Notebook name: {notebook_name}")
        
        # Test the notebook creation
        result = await notebook_client.create_notebook(
            workspace=workspace_id,
            notebook_name=notebook_name,
            content=notebook_content
        )
        
        print(f"Result: {result}")
        
        if isinstance(result, dict) and result.get("id"):
            print(f"✅ SUCCESS: Created notebook with ID: {result['id']}")
            return True
        else:
            print(f"❌ FAILED: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

async def test_workspace_resolution():
    """Test workspace name resolution"""
    try:
        credentials = get_azure_credentials("test-client-id", {})
        fabric_client = FabricApiClient(credentials)
        
        # Test workspace resolution
        workspace_name, workspace_id = await fabric_client.resolve_workspace_name_and_id("My workspace")
        print(f"✅ Workspace resolution: '{workspace_name}' -> {workspace_id}")
        return True
        
    except Exception as e:
        print(f"❌ Workspace resolution failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Fabric MCP Notebook Creation Fixes")
    print("=" * 50)
    
    # Test workspace resolution first
    print("\n1. Testing workspace resolution...")
    success1 = asyncio.run(test_workspace_resolution())
    
    # Test notebook creation
    print("\n2. Testing notebook creation...")
    success2 = asyncio.run(test_notebook_creation())
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
