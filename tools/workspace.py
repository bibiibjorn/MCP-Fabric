from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    WorkspaceClient,
)
from typing import Optional


@mcp.tool()
async def manage_workspace(
    action: str,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage Fabric workspaces.

    Args:
        action: Operation to perform:
            'list' - List all available workspaces
            'get_capacity' - Get capacity assignment for a workspace
        workspace: Workspace name or ID (required for get_capacity, uses context if not provided)
        ctx: Context object

    Returns:
        Workspace listing or capacity details
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        if action == "list":
            client = WorkspaceClient(fabric_client)
            return await client.list_workspaces()

        elif action == "get_capacity":
            ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
            if not ws:
                return "Workspace not set. Please set a workspace using set_context or provide workspace parameter."

            workspace_id = await fabric_client.resolve_workspace(ws)
            workspace_details = await fabric_client.get_workspace(workspace_id)

            if not workspace_details:
                return f"Could not retrieve details for workspace '{ws}'."

            capacity_id = workspace_details.get("capacityId", "N/A")
            workspace_name = workspace_details.get("displayName", ws)
            workspace_type = workspace_details.get("type", "N/A")

            markdown = f"# Workspace Capacity Assignment\n\n"
            markdown += f"**Workspace Name:** {workspace_name}\n"
            markdown += f"**Workspace ID:** {workspace_id}\n"
            markdown += f"**Workspace Type:** {workspace_type}\n"
            markdown += f"**Capacity ID:** {capacity_id}\n"

            return markdown

        else:
            return f"Error: Unknown action '{action}'. Use 'list' or 'get_capacity'."

    except Exception as e:
        return f"Error managing workspace: {str(e)}"
