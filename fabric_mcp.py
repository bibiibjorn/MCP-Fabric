from tools import (
    list_workspaces, get_workspace_capacity,
    list_warehouses, create_warehouse,
    list_lakehouses, create_lakehouse, list_shortcuts,
    list_tables, get_lakehouse_schema,
    list_semantic_models, get_semantic_model,
    list_reports, get_report,
    load_data_from_url,
    run_query, get_views,
    list_notebooks, create_notebook, get_notebook_content,
    update_notebook_cell, generate_notebook_code, validate_notebook_code,
    analyze_notebook_performance, list_spark_jobs, get_job_details,
    execute_dax_query, read_semantic_model_table, evaluate_measure,
    get_semantic_model_metadata, get_refresh_history,
    get_report_pages, get_report_details, get_report_definition,
    get_sql_endpoint,
    list_fabric_workloads, get_fabric_openapi_spec, get_fabric_platform_api,
    get_fabric_best_practices, list_fabric_best_practices,
    get_fabric_item_definition, list_fabric_item_definitions, get_fabric_api_examples,
    # Job Scheduler
    run_item_job, get_job_status, cancel_job, list_job_instances, manage_item_schedule,
    # Deployment Pipelines
    manage_deployment_pipeline, manage_deployment_stages, deploy_stage_content,
    # Environments
    manage_environment, manage_environment_compute, manage_environment_libraries,
    publish_environment,
)
from helpers.logging_config import get_logger
from helpers.utils.context import mcp, __ctx_cache
from helpers.utils.authentication import get_azure_credentials, ensure_authenticated, AuthenticationError
from helpers.clients import FabricApiClient, PowerBIClient
from mcp.server.fastmcp import Context
from typing import Optional
import uvicorn
import argparse
import sys

logger = get_logger(__name__)


@mcp.tool()
async def set_context(
    resource_type: str,
    resource: Optional[str] = None,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Set or clear session context for a specific resource.

    Sets the default workspace, lakehouse, warehouse, table, or semantic model
    for subsequent operations so you don't have to specify them every time.

    Args:
        resource_type: Type of resource to set:
            'workspace' - Set the active workspace
            'lakehouse' - Set the active lakehouse
            'warehouse' - Set the active warehouse
            'table' - Set the active table
            'semantic_model' - Set the active semantic model/dataset
            'clear' - Clear all session context
        resource: Name or ID of the resource (not needed for 'clear')
        workspace: Workspace name or ID (used when setting semantic_model to resolve it)
        ctx: Context object

    Returns:
        Confirmation message
    """
    try:
        if resource_type == "clear":
            __ctx_cache.clear()
            return "All session context cleared."

        if not resource:
            return f"Error: 'resource' parameter is required for resource_type '{resource_type}'."

        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        if resource_type == "workspace":
            workspace_name, workspace_id = await fabric_client.resolve_workspace_name_and_id(resource)
            __ctx_cache[f"{ctx.client_id}_workspace"] = workspace_id
            return f"Workspace set to '{workspace_name}' (ID: {workspace_id})."

        elif resource_type == "lakehouse":
            __ctx_cache[f"{ctx.client_id}_lakehouse"] = resource
            return f"Lakehouse set to '{resource}'."

        elif resource_type == "warehouse":
            __ctx_cache[f"{ctx.client_id}_warehouse"] = resource
            return f"Warehouse set to '{resource}'."

        elif resource_type == "table":
            __ctx_cache[f"{ctx.client_id}_table"] = resource
            return f"Table set to '{resource}'."

        elif resource_type == "semantic_model":
            powerbi_client = PowerBIClient(credential)

            workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
            if not workspace_id:
                return "Error: No workspace specified and no workspace context set. Set workspace first."
            workspace_id = await fabric_client.resolve_workspace(workspace_id)

            dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, resource, fabric_client)

            models = await fabric_client.get_semantic_models(workspace_id)
            model_name = next(
                (m.get("displayName") for m in models if m.get("id") == dataset_id),
                resource
            )

            __ctx_cache[f"{ctx.client_id}_semantic_model"] = dataset_id
            __ctx_cache[f"{ctx.client_id}_semantic_model_name"] = model_name

            if workspace:
                __ctx_cache[f"{ctx.client_id}_workspace"] = workspace_id

            return f"Semantic model set to '{model_name}' (ID: {dataset_id})."

        else:
            return f"Error: Unknown resource_type '{resource_type}'. Use: workspace, lakehouse, warehouse, table, semantic_model, or clear."

    except Exception as e:
        return f"Error setting context: {str(e)}"


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
