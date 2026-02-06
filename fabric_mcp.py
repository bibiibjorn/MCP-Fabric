from tools import (
    set_workspace, list_workspaces, get_workspace_capacity,
    set_warehouse, list_warehouses,
    set_lakehouse, list_lakehouses, list_shortcuts,
    set_table, list_tables, get_lakehouse_table_schema, get_all_lakehouse_schemas,
    list_semantic_models, get_semantic_model,
    list_reports, get_report,
    load_data_from_url,
    run_query, list_views, get_view_schema,
    list_notebooks, create_notebook, list_spark_jobs, get_job_details,
    list_fabric_workloads, get_fabric_openapi_spec, get_fabric_platform_api,
    get_fabric_best_practices, list_fabric_best_practices,
    get_fabric_item_definition, list_fabric_item_definitions, get_fabric_api_examples,
)
from helpers.logging_config import get_logger
from helpers.utils.context import mcp, __ctx_cache
from helpers.utils.authentication import ensure_authenticated, AuthenticationError
import uvicorn
import argparse
import sys

logger = get_logger(__name__)


@mcp.tool()
async def clear_context() -> str:
    """Clear the current session context.

    Returns:
        A string confirming the context has been cleared.
    """
    __ctx_cache.clear()
    return "Context cleared."


if __name__ == "__main__":
    # Initialize and run the server
    logger.info("Starting MCP server...")
    parser = argparse.ArgumentParser(description="Run MCP server")
    parser.add_argument("--port", type=int, default=None, help="Localhost port to listen on (HTTP mode)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--eager-auth", action="store_true", help="Authenticate at startup instead of on first tool use")
    args = parser.parse_args()

    # Validate port if provided
    if args.port is not None and not (1 <= args.port <= 65535):
        logger.error(f"Invalid port: {args.port}. Must be between 1 and 65535.")
        sys.exit(1)

    # By default, authentication happens lazily on first tool use
    # Use --eager-auth to authenticate at startup instead
    if args.eager_auth:
        try:
            ensure_authenticated()
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("Please ensure you have access to Microsoft Fabric.")
            sys.exit(1)

    if args.port:
        # Start the server with Streamable HTTP transport
        uvicorn.run(mcp.streamable_http_app, host=args.host, port=args.port)
    else:
        # Default to STDIO mode for Claude Desktop
        mcp.run(transport="stdio")
