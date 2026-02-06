from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    LakehouseClient,
)
from helpers.logging_config import get_logger
from typing import Optional

logger = get_logger(__name__)


@mcp.tool()
async def manage_lakehouse(
    action: str,
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage Fabric lakehouses.

    Args:
        action: Operation to perform:
            'list' - List all lakehouses in a workspace
            'create' - Create a new lakehouse
            'list_shortcuts' - List shortcuts (virtualized external data) in a lakehouse
        workspace: Workspace name or ID (optional, uses context if not provided)
        lakehouse: Lakehouse name or ID (required for 'list_shortcuts', uses context if not provided)
        name: Lakehouse name (required for 'create')
        description: Lakehouse description (optional for 'create')
        ctx: Context object

    Returns:
        Lakehouse listing, creation confirmation, or shortcuts listing
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        lakehouse_client = LakehouseClient(client=fabric_client)

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using set_context or provide workspace parameter."

        if action == "list":
            return await lakehouse_client.list_lakehouses(workspace=ws)

        elif action == "create":
            if not name:
                return "Error: 'name' is required for 'create' action."
            return await lakehouse_client.create_lakehouse(name=name, workspace=ws, description=description)

        elif action == "list_shortcuts":
            workspace_id = await fabric_client.resolve_workspace(ws)

            lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
            if not lh:
                return "Lakehouse not set. Please set a lakehouse using set_context or provide lakehouse parameter."

            lakehouse_id = await fabric_client.resolve_item_id(
                item=lh, type="Lakehouse", workspace=workspace_id
            )

            shortcuts = await lakehouse_client.list_shortcuts(workspace_id, lakehouse_id)

            if not shortcuts:
                return f"No shortcuts found in lakehouse '{lh}'."

            markdown = f"# Shortcuts in lakehouse '{lh}'\n\n"
            markdown += "| Name | Path | Target Location | Connection Type |\n"
            markdown += "|------|------|-----------------|----------------|\n"

            for shortcut in shortcuts:
                s_name = shortcut.get("name", "N/A")
                path = shortcut.get("path", "N/A")

                target = shortcut.get("target", {})
                if isinstance(target, dict):
                    if "oneLake" in target:
                        info = target["oneLake"]
                        target_location = f"OneLake: {info.get('workspaceName', 'N/A')}/{info.get('itemName', 'N/A')}{info.get('path', '')}"
                        connection_type = "OneLake"
                    elif "adlsGen2" in target:
                        info = target["adlsGen2"]
                        target_location = f"{info.get('endpoint', 'N/A')}{info.get('location', '')}"
                        connection_type = "ADLS Gen2"
                    elif "s3" in target:
                        info = target["s3"]
                        target_location = f"s3://{info.get('endpoint', 'N/A')}{info.get('location', '')}"
                        connection_type = "Amazon S3"
                    elif "googleCloudStorage" in target:
                        info = target["googleCloudStorage"]
                        target_location = f"gs://{info.get('endpoint', 'N/A')}{info.get('location', '')}"
                        connection_type = "GCS"
                    else:
                        target_location = str(target)
                        connection_type = "Unknown"
                else:
                    target_location = str(target)
                    connection_type = "Unknown"

                markdown += f"| {s_name} | {path} | {target_location} | {connection_type} |\n"

            markdown += f"\n**Total shortcuts:** {len(shortcuts)}"
            return markdown

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'create', or 'list_shortcuts'."

    except Exception as e:
        logger.error(f"Error managing lakehouse: {e}")
        return f"Error managing lakehouse: {e}"
