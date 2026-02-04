from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    TableClient,
    SQLClient,
    get_sql_endpoint,
)

from typing import Optional
from helpers.logging_config import get_logger

logger = get_logger(__name__)


@mcp.tool()
async def set_table(table_name: str, ctx: Context) -> str:
    """Set the current table for the session.

    Args:
        table_name: Name of the table to set
        ctx: Context object containing client information

    Returns:
        A string confirming the table has been set.
    """
    __ctx_cache[f"{ctx.client_id}_table"] = table_name
    return f"Table set to '{table_name}'."


@mcp.tool()
async def list_tables(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """List all tables in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace (optional)
        lakehouse: Name or ID of the lakehouse (optional)
        ctx: Context object containing client information

    Returns:
        A string containing the list of tables or an error message.
    """
    try:
        client = TableClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        tables = await client.list_tables(
            workspace_id=workspace
            if workspace
            else __ctx_cache[f"{ctx.client_id}_workspace"],
            rsc_id=lakehouse
            if lakehouse
            else __ctx_cache[f"{ctx.client_id}_lakehouse"],
        )

        return tables

    except Exception as e:
        return f"Error listing tables: {str(e)}"


@mcp.tool()
async def get_lakehouse_table_schema(
    workspace: Optional[str],
    lakehouse: Optional[str],
    table_name: str = None,
    ctx: Context = None,
) -> str:
    """Get schema for a specific table in a Fabric lakehouse.

    Args:
        workspace: Name or ID of the workspace
        lakehouse: Name or ID of the lakehouse
        table_name: Name of the table to retrieve
        ctx: Context object containing client information

    Returns:
        A string containing the schema of the specified table or an error message.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        client = TableClient(FabricApiClient(credential))

        if table_name is None:
            return "Table name must be specified."
        if lakehouse is None:
            if f"{ctx.client_id}_lakehouse" in __ctx_cache:
                lakehouse = __ctx_cache[f"{ctx.client_id}_lakehouse"]
            else:
                return "Lakehouse must be specified or set in the context."

        if workspace is None:
            if f"{ctx.client_id}_workspace" in __ctx_cache:
                workspace = __ctx_cache[f"{ctx.client_id}_workspace"]
            else:
                return "Workspace must be specified or set in the context."

        schema = await client.get_table_schema(
            workspace, lakehouse, "lakehouse", table_name, credential
        )

        return schema

    except Exception as e:
        return f"Error retrieving table schema: {str(e)}"


@mcp.tool()
async def get_all_lakehouse_schemas(
    lakehouse: Optional[str], workspace: Optional[str] = None, ctx: Context = None
) -> str:
    """Get schemas for all Delta tables in a Fabric lakehouse.

    Args:
        workspace: Name or ID of the workspace
        lakehouse: Name or ID of the lakehouse
        ctx: Context object containing client information

    Returns:
        A string containing the schemas of all Delta tables or an error message.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        client = TableClient(FabricApiClient(credential))

        if workspace is None:
            if f"{ctx.client_id}_workspace" in __ctx_cache:
                workspace = __ctx_cache[f"{ctx.client_id}_workspace"]
            else:
                return "Workspace must be specified or set in the context."
        if lakehouse is None:
            if f"{ctx.client_id}_lakehouse" in __ctx_cache:
                lakehouse = __ctx_cache[f"{ctx.client_id}_lakehouse"]
            else:
                return "Lakehouse must be specified or set in the context."
        schemas = await client.get_all_schemas(
            workspace, lakehouse, "lakehouse", credential
        )

        return schemas

    except Exception as e:
        return f"Error retrieving table schemas: {str(e)}"


@mcp.tool()
async def run_query(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    query: str = None,
    type: Optional[str] = None,  # Add type hint for 'type'
    ctx: Context = None,
) -> str:
    """Read data from a table in a warehouse or lakehouse.

    Args:
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        query: The SQL query to execute.
        type: Type of resource ('lakehouse' or 'warehouse'). If not provided, it will be inferred.
        ctx: Context object containing client information.
    Returns:
        A string confirming the data read or an error message.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        if query is None:
            raise ValueError("Query must be specified.")
        # Always resolve the SQL endpoint and database name
        database, sql_endpoint = await get_sql_endpoint(
            workspace=workspace,
            lakehouse=lakehouse,
            warehouse=warehouse,
            type=type,
        )
        if (
            not database
            or not sql_endpoint
            or sql_endpoint.startswith("Error")
            or sql_endpoint.startswith("No SQL endpoint")
        ):
            return f"Failed to resolve SQL endpoint: {sql_endpoint}"
        logger.info(f"Running query '{query}' on SQL endpoint {sql_endpoint}")
        client = SQLClient(sql_endpoint=sql_endpoint, database=database)
        df = client.run_query(query)
        if df.is_empty():
            return f"No data found for query '{query}'."

        # Convert to markdown for user-friendly display

        # markdown = f"### Query: {query} (shape: {df.shape})\n\n"
        # with pl.Config() as cfg:
        #     cfg.set_tbl_formatting('ASCII_MARKDOWN')
        #     display(Markdown(repr(df)))
        # markdown += f"\n\n### Data Preview:\n\n"
        # markdown += df.head(10).to_pandas().to_markdown(index=False)
        # markdown += f"\n\nColumns: {', '.join(df.columns)}"
        return df.to_dict()  # Return the DataFrame as a dictionary for easier handling
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return f"Error reading data: {str(e)}"


@mcp.tool()
async def list_views(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    schema_name: Optional[str] = None,
    type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """List all views in the SQL Analytics endpoint of a lakehouse or warehouse.

    Views are virtual tables created on top of Delta tables. They exist in the SQL
    Analytics endpoint and can be queried using SQL.

    Args:
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        schema_name: Filter views by schema name (e.g., 'dbo'). If not provided, lists all schemas.
        type: Type of resource ('lakehouse' or 'warehouse'). If not provided, it will be inferred.
        ctx: Context object containing client information.

    Returns:
        A list of views with their schema, name, and definition preview.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        # Resolve the SQL endpoint
        database, sql_endpoint = await get_sql_endpoint(
            workspace=workspace,
            lakehouse=lakehouse,
            warehouse=warehouse,
            type=type,
        )
        if (
            not database
            or not sql_endpoint
            or sql_endpoint.startswith("Error")
            or sql_endpoint.startswith("No SQL endpoint")
        ):
            return f"Failed to resolve SQL endpoint: {sql_endpoint}"

        logger.info(f"Listing views from SQL endpoint {sql_endpoint}")
        client = SQLClient(sql_endpoint=sql_endpoint, database=database)

        # Query INFORMATION_SCHEMA.VIEWS to get all views
        schema_filter = f"WHERE TABLE_SCHEMA = '{schema_name}'" if schema_name else ""
        query = f"""
            SELECT
                TABLE_SCHEMA as schema_name,
                TABLE_NAME as view_name,
                LEFT(VIEW_DEFINITION, 200) as definition_preview
            FROM INFORMATION_SCHEMA.VIEWS
            {schema_filter}
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """

        df = client.run_query(query)

        if df.is_empty():
            return f"No views found in {database}."

        return df.to_dict()

    except Exception as e:
        logger.error(f"Error listing views: {str(e)}")
        return f"Error listing views: {str(e)}"


@mcp.tool()
async def get_view_schema(
    view_name: str,
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    schema_name: str = "dbo",
    type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get the schema (columns and data types) of a specific view.

    Args:
        view_name: Name of the view to get schema for.
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        schema_name: Schema name where the view exists (default: 'dbo').
        type: Type of resource ('lakehouse' or 'warehouse'). If not provided, it will be inferred.
        ctx: Context object containing client information.

    Returns:
        The view's column definitions and full view definition.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        # Resolve the SQL endpoint
        database, sql_endpoint = await get_sql_endpoint(
            workspace=workspace,
            lakehouse=lakehouse,
            warehouse=warehouse,
            type=type,
        )
        if (
            not database
            or not sql_endpoint
            or sql_endpoint.startswith("Error")
            or sql_endpoint.startswith("No SQL endpoint")
        ):
            return f"Failed to resolve SQL endpoint: {sql_endpoint}"

        logger.info(f"Getting schema for view {schema_name}.{view_name}")
        client = SQLClient(sql_endpoint=sql_endpoint, database=database)

        # Get column information
        columns_query = f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema_name}'
              AND TABLE_NAME = '{view_name}'
            ORDER BY ORDINAL_POSITION
        """

        columns_df = client.run_query(columns_query)

        # Get view definition
        definition_query = f"""
            SELECT VIEW_DEFINITION
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA = '{schema_name}'
              AND TABLE_NAME = '{view_name}'
        """

        definition_df = client.run_query(definition_query)

        if columns_df.is_empty():
            return f"View '{schema_name}.{view_name}' not found."

        result = {
            "view_name": f"{schema_name}.{view_name}",
            "columns": columns_df.to_dict(),
            "definition": definition_df.to_dict() if not definition_df.is_empty() else None,
        }

        return result

    except Exception as e:
        logger.error(f"Error getting view schema: {str(e)}")
        return f"Error getting view schema: {str(e)}"
