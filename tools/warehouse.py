from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    WarehouseClient,
)
from typing import Optional


@mcp.tool()
async def manage_warehouse(
    action: str,
    workspace: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage Fabric warehouses.

    Args:
        action: Operation to perform:
            'list' - List all warehouses in a workspace
            'create' - Create a new warehouse
        workspace: Workspace name or ID (optional, uses context if not provided)
        name: Warehouse name (required for 'create')
        description: Warehouse description (optional for 'create')
        ctx: Context object

    Returns:
        Warehouse listing or creation confirmation
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        client = WarehouseClient(fabric_client)

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using set_context or provide workspace parameter."

        if action == "list":
            return await client.list_warehouses(ws)

        elif action == "create":
            if not name:
                return "Error: 'name' is required for 'create' action."
            response = await client.create_warehouse(name=name, workspace=ws, description=description)
            return f"Warehouse '{response.get('id', 'unknown')}' created successfully."

        else:
            return f"Error: Unknown action '{action}'. Use 'list' or 'create'."

    except Exception as e:
        return f"Error managing warehouse: {str(e)}"
