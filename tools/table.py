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
import re

logger = get_logger(__name__)


def _validate_sql_identifier(value: str) -> str:
    """Validate a SQL identifier (schema name, table name, view name).

    Only allows alphanumeric characters, underscores, spaces, hyphens, and dots.
    Raises ValueError if the identifier contains unsafe characters.
    """
    if not re.match(r'^[\w\s\-\.]+$', value):
        raise ValueError(f"Invalid SQL identifier: '{value}'.")
    return value


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

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
        if not ws:
            return "Workspace not set. Please set a workspace using 'set_workspace' command."
        if not lh:
            return "Lakehouse not set. Please set a lakehouse using 'set_lakehouse' command."

        tables = await client.list_tables(
            workspace_id=ws,
            rsc_id=lh,
        )

        return tables

    except Exception as e:
        return f"Error listing tables: {str(e)}"


@mcp.tool()
async def get_lakehouse_schema(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    table_name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get schema for one or all Delta tables in a Fabric lakehouse.

    If table_name is provided, returns the schema for that specific table.
    If table_name is omitted, returns schemas for all Delta tables.

    Args:
        workspace: Name or ID of the workspace (uses context if not provided)
        lakehouse: Name or ID of the lakehouse (uses context if not provided)
        table_name: Name of a specific table (optional - omit for all tables)
        ctx: Context object containing client information

    Returns:
        A string containing the table schema(s) or an error message.
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

        if table_name:
            # Single table schema
            schema = await client.get_table_schema(
                workspace, lakehouse, "lakehouse", table_name, credential
            )
            return schema
        else:
            # All table schemas
            schemas = await client.get_all_schemas(
                workspace, lakehouse, "lakehouse", credential
            )
            return schemas

    except Exception as e:
        return f"Error retrieving table schema: {str(e)}"


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
        return df.head(10).to_pandas().to_markdown(index=False)
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return f"Error reading data: {str(e)}"


@mcp.tool()
async def get_views(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    view_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """List views or get the schema of a specific view in a lakehouse or warehouse.

    If view_name is provided, returns the view's column definitions and full SQL definition.
    If view_name is omitted, lists all views with a definition preview.

    Args:
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        view_name: Name of a specific view to get schema for (optional - omit to list all).
        schema_name: Filter/scope by schema name (e.g., 'dbo'). Default 'dbo' when getting a specific view.
        type: Type of resource ('lakehouse' or 'warehouse'). If not provided, it will be inferred.
        ctx: Context object containing client information.

    Returns:
        A list of views or the detailed schema of a specific view.
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

        client = SQLClient(sql_endpoint=sql_endpoint, database=database)

        if view_name:
            # Get specific view schema
            effective_schema = schema_name or "dbo"
            _validate_sql_identifier(effective_schema)
            _validate_sql_identifier(view_name)

            logger.info(f"Getting schema for view {effective_schema}.{view_name}")

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
                WHERE TABLE_SCHEMA = '{effective_schema}'
                  AND TABLE_NAME = '{view_name}'
                ORDER BY ORDINAL_POSITION
            """

            columns_df = client.run_query(columns_query)

            # Get view definition
            definition_query = f"""
                SELECT VIEW_DEFINITION
                FROM INFORMATION_SCHEMA.VIEWS
                WHERE TABLE_SCHEMA = '{effective_schema}'
                  AND TABLE_NAME = '{view_name}'
            """

            definition_df = client.run_query(definition_query)

            if columns_df.is_empty():
                return f"View '{effective_schema}.{view_name}' not found."

            result = f"## View: {effective_schema}.{view_name}\n\n"
            result += "### Columns\n\n"
            result += columns_df.to_pandas().to_markdown(index=False)
            if not definition_df.is_empty():
                definition = definition_df.to_pandas().iloc[0, 0]
                result += f"\n\n### Definition\n\n```sql\n{definition}\n```"

            return result

        else:
            # List all views
            logger.info(f"Listing views from SQL endpoint {sql_endpoint}")

            if schema_name:
                _validate_sql_identifier(schema_name)
                schema_filter = f"WHERE TABLE_SCHEMA = '{schema_name}'"
            else:
                schema_filter = ""
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

            return df.to_pandas().to_markdown(index=False)

    except Exception as e:
        logger.error(f"Error with views: {str(e)}")
        return f"Error with views: {str(e)}"
