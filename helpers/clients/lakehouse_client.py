from helpers.utils import _is_valid_uuid
from helpers.logging_config import get_logger
from helpers.clients.fabric_client import FabricApiClient
from typing import Optional, Dict, Any, List
import base64
import json

logger = get_logger(__name__)


class LakehouseClient:
    def __init__(self, client: FabricApiClient):
        self.client = client

    async def list_lakehouses(self, workspace: str):
        """List all lakehouses in a workspace."""
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")
        lakehouses = await self.client.get_lakehouses(workspace)

        if not lakehouses:
            return f"No lakehouses found in workspace '{workspace}'."

        markdown = f"# Lakehouses in workspace '{workspace}'\n\n"
        markdown += "| ID | Name |\n"
        markdown += "|-----|------|\n"

        for lh in lakehouses:
            markdown += f"| {lh['id']} | {lh['displayName']} |\n"

        return markdown

    async def get_lakehouse(
        self,
        workspace: str,
        lakehouse: str,
    ) -> Optional[Dict[str, Any]]:
        """Get details of a specific lakehouse including SQL endpoint properties."""
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")

        if not lakehouse:
            raise ValueError("Lakehouse name cannot be empty.")

        response = await self.client.get_item(
            workspace_id=workspace, item_id=lakehouse, item_type="Lakehouse"
        )
        logger.info(f"Lakehouse details: {response}")
        return response

    async def resolve_lakehouse(self, workspace_id: str, lakehouse_name: str):
        """Resolve lakehouse name to lakehouse ID."""
        return await self.client.resolve_item_name_and_id(
            workspace=workspace_id, item=lakehouse_name, type="Lakehouse"
        )

    async def create_lakehouse(
        self,
        name: str,
        workspace: str,
        description: Optional[str] = None,
    ):
        """Create a new lakehouse."""
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")

        if not name:
            raise ValueError("Lakehouse name cannot be empty.")

        return await self.client.create_item(
            name=name, workspace=workspace, description="description", type="Lakehouse"
        )

    async def get_lakehouse_definition(
        self, workspace_id: str, lakehouse_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get lakehouse definition including shortcuts.

        The definition contains all configuration files including shortcuts.metadata.json.

        Args:
            workspace_id: Workspace ID (UUID)
            lakehouse_id: Lakehouse ID (UUID)

        Returns:
            Lakehouse definition with parts array containing file contents
        """
        if not _is_valid_uuid(workspace_id):
            raise ValueError("Invalid workspace ID.")
        if not _is_valid_uuid(lakehouse_id):
            raise ValueError("Invalid lakehouse ID.")

        endpoint = f"workspaces/{workspace_id}/lakehouses/{lakehouse_id}/getDefinition"
        logger.info(f"Getting lakehouse definition for {lakehouse_id}")

        definition = await self.client._make_request(
            endpoint=endpoint, method="POST", lro=True, lro_poll_interval=2
        )

        return definition

    async def list_shortcuts(
        self, workspace_id: str, lakehouse_id: str
    ) -> List[Dict[str, Any]]:
        """Extract shortcuts from lakehouse definition.

        Shortcuts are stored in shortcuts.metadata.json within the lakehouse definition.

        Args:
            workspace_id: Workspace ID (UUID)
            lakehouse_id: Lakehouse ID (UUID)

        Returns:
            List of shortcut objects with name, path, target, and type information
        """
        definition = await self.get_lakehouse_definition(workspace_id, lakehouse_id)

        if not definition:
            logger.warning(f"No definition returned for lakehouse {lakehouse_id}")
            return []

        # Parse definition parts to find shortcuts.metadata.json
        shortcuts = []
        parts = definition.get("definition", {}).get("parts", [])

        for part in parts:
            path = part.get("path", "")
            if "shortcuts" in path.lower() and path.endswith(".json"):
                try:
                    # Decode base64 payload
                    payload = part.get("payload", "")
                    if not payload:
                        continue

                    decoded_content = base64.b64decode(payload).decode("utf-8")
                    shortcuts_data = json.loads(decoded_content)

                    # Extract shortcuts array
                    if isinstance(shortcuts_data, dict):
                        shortcuts = shortcuts_data.get("shortcuts", [])
                    elif isinstance(shortcuts_data, list):
                        shortcuts = shortcuts_data

                    logger.info(f"Found {len(shortcuts)} shortcuts in lakehouse")
                    break
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Error parsing shortcuts file: {str(e)}")
                    continue

        return shortcuts
