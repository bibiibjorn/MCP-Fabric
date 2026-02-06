from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    WorkspaceClient,
)
from typing import Optional


@mcp.tool()
async def list_workspaces(ctx: Context) -> str:
    """List all available Fabric workspaces.

    Args:
        ctx: Context object containing client information

    Returns:
        A string containing the list of workspaces or an error message.
    """
    try:
        client = WorkspaceClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        workspaces = await client.list_workspaces()

        return workspaces

    except Exception as e:
        return f"Error listing workspaces: {str(e)}"


@mcp.tool()
async def get_workspace_capacity(
    workspace: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get capacity assignment for a workspace.

    Retrieves the capacity ID assigned to a workspace. The capacity determines
    the compute resources available to the workspace.

    Args:
        workspace: Workspace name or ID (optional - uses context if not provided)
        ctx: Context object containing client information

    Returns:
        Capacity ID and workspace details
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))

        # Resolve workspace
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using 'set_workspace' command."

        workspace_id = await fabric_client.resolve_workspace(ws)

        # Get workspace details which include capacity information
        workspace_details = await fabric_client.get_workspace(workspace_id)

        if not workspace_details:
            return f"Could not retrieve details for workspace '{ws}'."

        # Extract capacity information
        capacity_id = workspace_details.get("capacityId", "N/A")
        workspace_name = workspace_details.get("displayName", ws)
        workspace_type = workspace_details.get("type", "N/A")

        # Format output
        markdown = f"# Workspace Capacity Assignment\n\n"
        markdown += f"**Workspace Name:** {workspace_name}\n"
        markdown += f"**Workspace ID:** {workspace_id}\n"
        markdown += f"**Workspace Type:** {workspace_type}\n"
        markdown += f"**Capacity ID:** {capacity_id}\n\n"

        if capacity_id == "N/A" or not capacity_id:
            markdown += "⚠️ **Note:** This workspace is not assigned to a capacity. "
            markdown += "Workspaces must be assigned to a Fabric capacity to use most Fabric features.\n"
        else:
            markdown += "✅ **Status:** Workspace is assigned to a Fabric capacity.\n\n"
            markdown += "### About Capacities:\n"
            markdown += "- Capacities provide compute and storage resources for Fabric workspaces\n"
            markdown += "- Each capacity has a specific SKU (F2, F4, F8, etc.) determining available resources\n"
            markdown += "- Multiple workspaces can share a single capacity\n"
            markdown += "- Capacity usage affects performance and throttling\n"

        return markdown

    except Exception as e:
        return f"Error getting workspace capacity: {str(e)}"
