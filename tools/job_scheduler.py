from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import FabricApiClient
from helpers.logging_config import get_logger
from typing import Optional, Dict, Any

logger = get_logger(__name__)

# Default job type mappings per item type
_JOB_TYPE_DEFAULTS = {
    "Notebook": "RunNotebook",
    "DataPipeline": "Pipeline",
    "SparkJobDefinition": "SparkJobDefinition",
    "SemanticModel": "DefaultJob",
    "Lakehouse": "DefaultJob",
    "Warehouse": "DefaultJob",
}

# Item type to API plural path segment
_ITEM_TYPE_PATHS = {
    "Notebook": "notebooks",
    "DataPipeline": "dataPipelines",
    "SparkJobDefinition": "sparkJobDefinitions",
    "SemanticModel": "semanticModels",
    "Lakehouse": "lakehouses",
    "Warehouse": "warehouses",
}


async def _resolve_workspace_and_item(
    fabric_client: FabricApiClient,
    workspace: Optional[str],
    item: str,
    item_type: str,
    client_id: str,
) -> tuple:
    """Resolve workspace ID and item ID for job operations."""
    ws = workspace or __ctx_cache.get(f"{client_id}_workspace")
    if not ws:
        raise ValueError("Workspace not set. Use set_context('workspace', ...) or provide workspace parameter.")
    workspace_id = await fabric_client.resolve_workspace(ws)
    item_id = await fabric_client.resolve_item_id(
        item=item, type=item_type, workspace=workspace_id
    )
    return workspace_id, item_id


@mcp.tool()
async def run_item_job(
    item: str,
    item_type: str,
    job_type: Optional[str] = None,
    workspace: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    ctx: Context = None,
) -> str:
    """Run an on-demand job for any Fabric item.

    Supports running notebooks, data pipelines, Spark job definitions,
    and triggering semantic model refreshes.

    Args:
        item: Item name or ID
        item_type: Type of item: Notebook, DataPipeline, SparkJobDefinition, SemanticModel
        job_type: Job type override (auto-detected from item_type if not provided)
        workspace: Workspace name or ID (uses context if not provided)
        parameters: Optional job parameters dict (passed as executionData.parameters)
        ctx: Context object

    Returns:
        Job instance details with ID and status
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        resolved_job_type = job_type or _JOB_TYPE_DEFAULTS.get(item_type, "DefaultJob")

        body = {}
        if parameters:
            body["executionData"] = {"parameters": parameters}

        endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances?jobType={resolved_job_type}"

        response = await fabric_client._make_request(
            endpoint=endpoint,
            method="POST",
            params=body,
        )

        # The API returns 202 with Location header containing the job instance URL
        # _make_request may return None for 202 without LRO polling
        if response and isinstance(response, dict):
            instance_id = response.get("id", "N/A")
            status = response.get("status", "Accepted")
        else:
            instance_id = "Submitted"
            status = "Accepted"

        __ctx_cache[f"{ctx.client_id}_last_job_instance"] = instance_id

        markdown = f"# Job Started\n\n"
        markdown += f"**Item:** {item} ({item_type})\n"
        markdown += f"**Job Type:** {resolved_job_type}\n"
        markdown += f"**Instance ID:** {instance_id}\n"
        markdown += f"**Status:** {status}\n\n"
        if parameters:
            markdown += f"**Parameters:** {parameters}\n\n"
        markdown += "Use `get_job_status` to check progress or `cancel_job` to cancel."

        return markdown

    except Exception as e:
        return f"Error running job: {str(e)}"


@mcp.tool()
async def get_job_status(
    item: str,
    item_type: str,
    job_instance_id: str,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get the status of a specific job instance.

    Args:
        item: Item name or ID
        item_type: Type of item (Notebook, DataPipeline, SparkJobDefinition, SemanticModel)
        job_instance_id: ID of the job instance to check
        workspace: Workspace name or ID (uses context if not provided)
        ctx: Context object

    Returns:
        Job status details including state, start/end times, and failure info
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}"
        result = await fabric_client._make_request(endpoint=endpoint)

        if not result:
            return f"Error: Could not retrieve job status for instance '{job_instance_id}'."

        status = result.get("status", "Unknown")
        start_time = result.get("startTimeUtc", "N/A")
        end_time = result.get("endTimeUtc", "N/A")
        invoke_type = result.get("invokeType", "N/A")
        failure_reason = result.get("failureReason", None)

        markdown = f"# Job Status\n\n"
        markdown += f"**Instance ID:** {job_instance_id}\n"
        markdown += f"**Status:** {status}\n"
        markdown += f"**Invoke Type:** {invoke_type}\n"
        markdown += f"**Start Time:** {start_time}\n"
        markdown += f"**End Time:** {end_time}\n"

        if failure_reason:
            markdown += f"\n**Failure Reason:** {failure_reason.get('message', 'Unknown')}\n"
            error_code = failure_reason.get('errorCode', '')
            if error_code:
                markdown += f"**Error Code:** {error_code}\n"

        return markdown

    except Exception as e:
        return f"Error getting job status: {str(e)}"


@mcp.tool()
async def cancel_job(
    item: str,
    item_type: str,
    job_instance_id: str,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Cancel a running job instance.

    Args:
        item: Item name or ID
        item_type: Type of item (Notebook, DataPipeline, SparkJobDefinition, SemanticModel)
        job_instance_id: ID of the job instance to cancel
        workspace: Workspace name or ID (uses context if not provided)
        ctx: Context object

    Returns:
        Confirmation or error message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}/cancel"
        await fabric_client._make_request(
            endpoint=endpoint,
            method="POST",
            params={},
        )

        return f"Job instance '{job_instance_id}' cancellation requested successfully."

    except Exception as e:
        return f"Error cancelling job: {str(e)}"


@mcp.tool()
async def list_job_instances(
    item: str,
    item_type: str,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """List recent job instances for a Fabric item.

    Args:
        item: Item name or ID
        item_type: Type of item (Notebook, DataPipeline, SparkJobDefinition, SemanticModel)
        workspace: Workspace name or ID (uses context if not provided)
        ctx: Context object

    Returns:
        Table of recent job runs with status, times, and types
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances"
        results = await fabric_client._make_request(
            endpoint=endpoint,
            use_pagination=True,
        )

        if not results:
            return f"No job instances found for '{item}'."

        markdown = f"# Job Instances for {item}\n\n"
        markdown += "| Instance ID | Status | Job Type | Start Time | End Time |\n"
        markdown += "|---|---|---|---|---|\n"

        for job in results:
            instance_id = job.get("id", "N/A")
            status = job.get("status", "N/A")
            job_type = job.get("jobType", "N/A")
            start_time = job.get("startTimeUtc", "N/A")
            end_time = job.get("endTimeUtc", "N/A")
            markdown += f"| {instance_id} | {status} | {job_type} | {start_time} | {end_time} |\n"

        markdown += f"\n**Total:** {len(results)} instance(s)"
        return markdown

    except Exception as e:
        return f"Error listing job instances: {str(e)}"


@mcp.tool()
async def manage_item_schedule(
    action: str,
    item: str,
    item_type: str,
    job_type: Optional[str] = None,
    workspace: Optional[str] = None,
    schedule_id: Optional[str] = None,
    schedule_config: Optional[Dict[str, Any]] = None,
    ctx: Context = None,
) -> str:
    """Create, list, update, or delete job schedules for a Fabric item.

    Args:
        action: Operation: 'create', 'list', 'update', 'delete'
        item: Item name or ID
        item_type: Type of item (Notebook, DataPipeline, SparkJobDefinition, SemanticModel)
        job_type: Job type for the schedule (auto-detected from item_type if not provided)
        workspace: Workspace name or ID (uses context if not provided)
        schedule_id: Schedule ID (required for update/delete)
        schedule_config: Schedule configuration dict (required for create/update).
            Example: {"enabled": true, "configuration": {"startDateTime": "2025-01-01T00:00:00Z", "type": "Daily", "interval": 1}}
        ctx: Context object

    Returns:
        Schedule details or confirmation message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        resolved_job_type = job_type or _JOB_TYPE_DEFAULTS.get(item_type, "DefaultJob")
        base_endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/{resolved_job_type}/schedules"

        if action == "list":
            results = await fabric_client._make_request(
                endpoint=base_endpoint,
                use_pagination=True,
            )

            if not results:
                return f"No schedules found for '{item}'."

            markdown = f"# Schedules for {item}\n\n"
            markdown += "| Schedule ID | Enabled | Type | Start | End |\n"
            markdown += "|---|---|---|---|---|\n"

            for schedule in results:
                sid = schedule.get("id", "N/A")
                enabled = schedule.get("enabled", False)
                config = schedule.get("configuration", {})
                stype = config.get("type", "N/A")
                start = config.get("startDateTime", "N/A")
                end = config.get("endDateTime", "N/A")
                markdown += f"| {sid} | {enabled} | {stype} | {start} | {end} |\n"

            return markdown

        elif action == "create":
            if not schedule_config:
                return "Error: schedule_config is required for 'create' action."

            result = await fabric_client._make_request(
                endpoint=base_endpoint,
                method="POST",
                params=schedule_config,
            )

            if result and isinstance(result, dict):
                return f"Schedule created successfully. ID: {result.get('id', 'N/A')}"
            return "Schedule creation submitted."

        elif action == "update":
            if not schedule_id:
                return "Error: schedule_id is required for 'update' action."
            if not schedule_config:
                return "Error: schedule_config is required for 'update' action."

            endpoint = f"{base_endpoint}/{schedule_id}"
            result = await fabric_client._make_request(
                endpoint=endpoint,
                method="PATCH",
                params=schedule_config,
            )

            return f"Schedule '{schedule_id}' updated successfully."

        elif action == "delete":
            if not schedule_id:
                return "Error: schedule_id is required for 'delete' action."

            endpoint = f"{base_endpoint}/{schedule_id}"
            await fabric_client._make_request(
                endpoint=endpoint,
                method="DELETE",
            )

            return f"Schedule '{schedule_id}' deleted successfully."

        else:
            return f"Error: Unknown action '{action}'. Use 'create', 'list', 'update', or 'delete'."

    except Exception as e:
        return f"Error managing schedule: {str(e)}"
