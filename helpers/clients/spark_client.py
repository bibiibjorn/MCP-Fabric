"""
Spark Client for managing Spark job executions (Livy sessions) in Microsoft Fabric.
"""

from typing import Optional, Dict, Any, List
from helpers.clients.fabric_client import FabricApiClient
from helpers.logging_config import get_logger

logger = get_logger(__name__)

# Cache for ID-to-name lookups (shared across instances)
_capacity_cache: Dict[str, str] = {}
_workspace_cache: Dict[str, str] = {}


class SparkClient:
    """Client for Spark/Livy session operations in Microsoft Fabric."""

    def __init__(self, client: FabricApiClient):
        """
        Initialize SparkClient.

        Args:
            client: Initialized FabricApiClient instance
        """
        self.client = client
        self._capacities_loaded = False
        self._workspaces_loaded = False

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

        # Extract state filter for client-side filtering (API doesn't reliably filter by state)
        state_filter = filters.pop("state", None) if filters else None
        params = filters or {}

        sessions = await self.client._make_request(
            endpoint=endpoint, use_pagination=True, data_key="value", params=params
        )

        sessions = sessions if sessions else []

        # Apply client-side state filtering since the API doesn't reliably support it
        if state_filter and sessions:
            sessions = [s for s in sessions if s.get("state") == state_filter]

        return sessions

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

    async def get_job_instance(
        self, workspace_id: str, item_id: str, job_instance_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a job instance (execution details).

        Args:
            workspace_id: Workspace ID (UUID)
            item_id: Item ID (Notebook, Spark Job Definition, etc.)
            job_instance_id: Job instance ID from Livy session

        Returns:
            Job instance details including failure information or None if not found
        """
        endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}"
        logger.info(f"Getting job instance details for {job_instance_id}")

        try:
            job_instance = await self.client._make_request(endpoint=endpoint)
            return job_instance
        except Exception as e:
            logger.warning(f"Failed to get job instance details: {str(e)}")
            return None

    async def get_capacities(self) -> List[Dict[str, Any]]:
        """
        Get list of all capacities accessible to the user.

        Returns:
            List of capacity objects with id, displayName, sku, region, state
        """
        endpoint = "capacities"
        logger.info("Fetching capacities list")

        try:
            capacities = await self.client._make_request(
                endpoint=endpoint, use_pagination=True, data_key="value"
            )
            return capacities if capacities else []
        except Exception as e:
            logger.warning(f"Failed to get capacities: {str(e)}")
            return []

    async def resolve_capacity_name(self, capacity_id: str) -> str:
        """
        Resolve a capacity ID to its display name.

        Args:
            capacity_id: Capacity ID (UUID)

        Returns:
            Capacity display name or the original ID if not found
        """
        global _capacity_cache

        if not capacity_id:
            return "N/A"

        # Check cache first
        if capacity_id in _capacity_cache:
            return _capacity_cache[capacity_id]

        # Load all capacities if not yet loaded
        if not self._capacities_loaded:
            try:
                capacities = await self.get_capacities()
                for cap in capacities:
                    cap_id = cap.get("id")
                    cap_name = cap.get("displayName", cap.get("name"))
                    if cap_id and cap_name:
                        _capacity_cache[cap_id] = cap_name
                self._capacities_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load capacities for name resolution: {str(e)}")

        return _capacity_cache.get(capacity_id, capacity_id)

    async def resolve_workspace_name(self, workspace_id: str) -> str:
        """
        Resolve a workspace ID to its display name.

        Args:
            workspace_id: Workspace ID (UUID)

        Returns:
            Workspace display name or the original ID if not found
        """
        global _workspace_cache

        if not workspace_id:
            return "N/A"

        # Check cache first
        if workspace_id in _workspace_cache:
            return _workspace_cache[workspace_id]

        # Load all workspaces if not yet loaded
        if not self._workspaces_loaded:
            try:
                workspaces = await self.client.get_workspaces()
                for ws in workspaces:
                    ws_id = ws.get("id")
                    ws_name = ws.get("displayName")
                    if ws_id and ws_name:
                        _workspace_cache[ws_id] = ws_name
                self._workspaces_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load workspaces for name resolution: {str(e)}")

        return _workspace_cache.get(workspace_id, workspace_id)

    async def enrich_session_with_names(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a session object by resolving IDs to human-readable names.

        Resolves:
        - capacityId -> capacityName
        - workspaceId -> workspaceName
        - item.id -> itemName (if not already present)

        Args:
            session: Raw session dict from API

        Returns:
            Enriched session dict with additional *Name fields
        """
        if not session:
            return session

        enriched = session.copy()

        # Resolve capacity name
        capacity_id = session.get("capacityId")
        if capacity_id:
            enriched["capacityName"] = await self.resolve_capacity_name(capacity_id)

        # Resolve workspace name (if workspaceId is present)
        workspace_id = session.get("workspaceId")
        if workspace_id:
            enriched["workspaceName"] = await self.resolve_workspace_name(workspace_id)

        return enriched
