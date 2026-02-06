from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    TableClient,
    SQLClient,
    get_sql_endpoint as _resolve_sql_endpoint,
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
async def manage_tables(
    action: str,
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    table_name: Optional[str] = None,
    view_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage tables, schemas, views, and SQL endpoints in Fabric lakehouses/warehouses.

    Args:
        action: Operation to perform:
            'list' - List all tables in a lakehouse
            'get_schema' - Get schema for one or all Delta tables
            'get_views' - List views or get a specific view's schema and SQL definition
            'get_sql_endpoint' - Get the SQL endpoint connection string
        workspace: Workspace name or ID (uses context if not provided)
        lakehouse: Lakehouse name or ID (uses context if not provided)
        warehouse: Warehouse name or ID (uses context if not provided)
        table_name: Table name (for 'get_schema' to get a single table)
        view_name: View name (for 'get_views' to get a specific view)
        schema_name: SQL schema name e.g. 'dbo' (for 'get_views')
        type: Resource type ('lakehouse' or 'warehouse'). Inferred if not provided.
        ctx: Context object

    Returns:
        Table listings, schema details, view definitions, or SQL endpoint info
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        credential = get_azure_credentials(ctx.client_id, __ctx_cache)

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using set_context or provide workspace parameter."

        if action == "list":
            client = TableClient(FabricApiClient(credential))
            lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
            if not lh:
                return "Lakehouse not set. Please set a lakehouse using set_context or provide lakehouse parameter."
            return await client.list_tables(workspace_id=ws, rsc_id=lh)

        elif action == "get_schema":
            client = TableClient(FabricApiClient(credential))
            lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
            if not lh:
                return "Lakehouse not set. Please set a lakehouse using set_context or provide lakehouse parameter."

            if table_name:
                return await client.get_table_schema(ws, lh, "lakehouse", table_name, credential)
            else:
                return await client.get_all_schemas(ws, lh, "lakehouse", credential)

        elif action == "get_views":
            lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
            wh = warehouse or __ctx_cache.get(f"{ctx.client_id}_warehouse")

            database, sql_endpoint = await _resolve_sql_endpoint(
                workspace=ws, lakehouse=lh, warehouse=wh, type=type,
            )
            if not database or not sql_endpoint or sql_endpoint.startswith("Error") or sql_endpoint.startswith("No SQL endpoint"):
                return f"Failed to resolve SQL endpoint: {sql_endpoint}"

            client = SQLClient(sql_endpoint=sql_endpoint, database=database)

            if view_name:
                effective_schema = schema_name or "dbo"
                _validate_sql_identifier(effective_schema)
                _validate_sql_identifier(view_name)

                columns_query = f"""
                    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH,
                           NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{effective_schema}' AND TABLE_NAME = '{view_name}'
                    ORDER BY ORDINAL_POSITION
                """
                columns_df = client.run_query(columns_query)

                definition_query = f"""
                    SELECT VIEW_DEFINITION FROM INFORMATION_SCHEMA.VIEWS
                    WHERE TABLE_SCHEMA = '{effective_schema}' AND TABLE_NAME = '{view_name}'
                """
                definition_df = client.run_query(definition_query)

                if columns_df.is_empty():
                    return f"View '{effective_schema}.{view_name}' not found."

                result = f"## View: {effective_schema}.{view_name}\n\n### Columns\n\n"
                result += columns_df.to_pandas().to_markdown(index=False)
                if not definition_df.is_empty():
                    definition = definition_df.to_pandas().iloc[0, 0]
                    result += f"\n\n### Definition\n\n```sql\n{definition}\n```"
                return result
            else:
                if schema_name:
                    _validate_sql_identifier(schema_name)
                    schema_filter = f"WHERE TABLE_SCHEMA = '{schema_name}'"
                else:
                    schema_filter = ""

                query = f"""
                    SELECT TABLE_SCHEMA as schema_name, TABLE_NAME as view_name,
                           LEFT(VIEW_DEFINITION, 200) as definition_preview
                    FROM INFORMATION_SCHEMA.VIEWS {schema_filter}
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                """
                df = client.run_query(query)
                if df.is_empty():
                    return f"No views found in {database}."
                return df.to_pandas().to_markdown(index=False)

        elif action == "get_sql_endpoint":
            lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
            wh = warehouse or __ctx_cache.get(f"{ctx.client_id}_warehouse")
            if not lh and not wh:
                return "Either lakehouse or warehouse must be specified or set in context."

            name, endpoint = await _resolve_sql_endpoint(
                workspace=ws, lakehouse=lh, warehouse=wh, type=type,
            )
            return endpoint if endpoint else f"No SQL endpoint found."

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'get_schema', 'get_views', or 'get_sql_endpoint'."

    except Exception as e:
        logger.error(f"Error managing tables: {str(e)}")
        return f"Error managing tables: {str(e)}"


@mcp.tool()
async def run_query(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    query: str = None,
    type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Execute a SQL query against a Fabric warehouse or lakehouse.

    Args:
        workspace: Workspace name or ID (optional).
        lakehouse: Lakehouse name or ID (optional).
        warehouse: Warehouse name or ID (optional).
        query: The SQL query to execute.
        type: Resource type ('lakehouse' or 'warehouse'). Inferred if not provided.
        ctx: Context object.

    Returns:
        Query results as a markdown table (first 10 rows).
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        if query is None:
            raise ValueError("Query must be specified.")

        database, sql_endpoint = await _resolve_sql_endpoint(
            workspace=workspace, lakehouse=lakehouse, warehouse=warehouse, type=type,
        )
        if not database or not sql_endpoint or sql_endpoint.startswith("Error") or sql_endpoint.startswith("No SQL endpoint"):
            return f"Failed to resolve SQL endpoint: {sql_endpoint}"

        logger.info(f"Running query '{query}' on SQL endpoint {sql_endpoint}")
        client = SQLClient(sql_endpoint=sql_endpoint, database=database)
        df = client.run_query(query)
        if df.is_empty():
            return f"No data found for query '{query}'."

        return df.head(10).to_pandas().to_markdown(index=False)
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return f"Error reading data: {str(e)}"
