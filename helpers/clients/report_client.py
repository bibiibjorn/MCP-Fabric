from helpers.logging_config import get_logger
from helpers.clients.fabric_client import FabricApiClient

logger = get_logger(__name__)


class ReportClient:
    def __init__(self, client: FabricApiClient):
        self.client = client

    async def list_reports(self, workspace_id: str):
        """List all reports in a workspace.

        Args:
            workspace_id: ID of the workspace (must be resolved to UUID first)

        Returns:
            List of report dictionaries, or empty list if none found
        """
        # Resolve workspace name to ID if needed
        workspace_id = await self.client.resolve_workspace(workspace_id)

        reports = await self.client.get_reports(workspace_id)

        # Always return a list, even if empty
        if not reports:
            return []

        return reports

    async def get_report(self, workspace_id: str, report_id: str) -> dict:
        """Get a specific report by ID.

        Args:
            workspace_id: ID of the workspace (must be resolved to UUID first)
            report_id: ID of the report

        Returns:
            Report dictionary, or None if not found
        """
        # Resolve workspace name to ID if needed
        workspace_id = await self.client.resolve_workspace(workspace_id)

        report = await self.client.get_report(workspace_id, report_id)

        # Return None instead of string message for not found
        if not report:
            return None

        return report
