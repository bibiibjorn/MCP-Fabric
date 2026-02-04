"""
Spark Client for managing Spark job executions (Livy sessions) in Microsoft Fabric.
"""

from typing import Optional, Dict, Any, List
from helpers.clients.fabric_client import FabricApiClient
from helpers.logging_config import get_logger

logger = get_logger(__name__)


class SparkClient:
    """Client for Spark/Livy session operations in Microsoft Fabric."""

    def __init__(self, client: FabricApiClient):
        """
        Initialize SparkClient.

        Args:
            client: Initialized FabricApiClient instance
        """
        self.client = client

    async def list_notebook_sessions(
        self, workspace_id: str, notebook_id: str
    ) -> List[Dict[str, Any]]:
        """
        List Livy sessions for a specific notebook.

        Args:
            workspace_id: Workspace ID (UUID)
            notebook_id: Notebook ID (UUID)

        Returns:
            List of Livy session objects
        """
        endpoint = f"workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions"
        logger.info(f"Listing Livy sessions for notebook {notebook_id}")

        sessions = await self.client._make_request(
            endpoint=endpoint, use_pagination=True, data_key="value"
        )

        return sessions if sessions else []

    async def list_workspace_sessions(
        self, workspace_id: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List all Livy sessions in a workspace with optional filters.

        Uses beta API endpoint for enhanced filtering capabilities.

        Args:
            workspace_id: Workspace ID (UUID)
            filters: Optional filters (state, submittedDateTime, endDateTime, submitter.Id)

        Returns:
            List of Livy session objects
        """
        # Use beta endpoint for filtering support
        endpoint = f"workspaces/{workspace_id}/spark/livySessions?beta=true"
        logger.info(f"Listing all Livy sessions in workspace {workspace_id}")

        params = filters or {}

        sessions = await self.client._make_request(
            endpoint=endpoint, use_pagination=True, data_key="value", params=params
        )

        return sessions if sessions else []

    async def get_session_details(
        self, workspace_id: str, notebook_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific Livy session.

        Args:
            workspace_id: Workspace ID (UUID)
            notebook_id: Notebook ID (UUID)
            session_id: Livy session ID

        Returns:
            Session details or None if not found
        """
        endpoint = (
            f"workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions/{session_id}"
        )
        logger.info(f"Getting details for Livy session {session_id}")

        session = await self.client._make_request(endpoint=endpoint)

        return session
