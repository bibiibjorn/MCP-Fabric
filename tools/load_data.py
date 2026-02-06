from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    LakehouseClient,
    WarehouseClient,
    get_sql_endpoint,
)
from helpers.logging_config import get_logger
import tempfile
import os
import httpx
from typing import Optional

logger = get_logger(__name__)


@mcp.tool()
async def load_data(
    url: str,
    destination_table: str,
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Load data from a URL into a table in a warehouse or lakehouse.

    Args:
        url: The URL to download data from (CSV or Parquet supported).
        destination_table: The name of the table to load data into.
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        ctx: Context object containing client information.
    Returns:
        A string confirming the data load or an error message.
    """
    try:
        # Download the file
        response = httpx.get(url)
        if response.status_code != 200:
            return f"Failed to download file from URL: {url}"
        file_ext = url.split("?")[0].split(".")[-1].lower()
        if file_ext not in ("csv", "parquet"):
            return f"Unsupported file type: {file_ext}. Only CSV and Parquet are supported."
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_ext}"
        ) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        # Choose destination: lakehouse or warehouse
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        resource_id = None
        resource_type = None
        if lakehouse:
            client = LakehouseClient(FabricApiClient(credential))
            resource_id = lakehouse
            resource_type = "lakehouse"
        elif warehouse:
            client = WarehouseClient(FabricApiClient(credential))
            resource_id = warehouse
            resource_type = "warehouse"
        else:
            return "Either lakehouse or warehouse must be specified."
        # TODO: Implement actual data loading via client API
        os.remove(tmp_path)
        return f"Error: load_data is not yet implemented. The file was downloaded from {url} ({file_ext}) but data loading into {resource_type} '{resource_id}' is not supported yet. Use 'run_query' or PySpark notebooks to load data instead."
    except Exception as e:
        return f"Error loading data: {str(e)}"


# @mcp.resource(
#         uri="tables://{table_name}",
# )
