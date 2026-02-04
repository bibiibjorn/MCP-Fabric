from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    WorkspaceClient,
)


@mcp.tool()
async def set_workspace(workspace: str, ctx: Context) -> str:
    """Set the current workspace for the session.

    Args:
        workspace: Name or ID of the workspace
        ctx: Context object containing client information
    Returns:
        A string confirming the workspace has been set.
    """
    try:
        # Resolve workspace name to ID (GUID) before storing
        fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        workspace_name, workspace_id = await fabric_client.resolve_workspace_name_and_id(workspace)

        # Store the resolved GUID, not the name
        __ctx_cache[f"{ctx.client_id}_workspace"] = workspace_id
        return f"Workspace set to '{workspace_name}' (ID: {workspace_id})."
    except Exception as e:
        return f"Error setting workspace: {str(e)}"


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
