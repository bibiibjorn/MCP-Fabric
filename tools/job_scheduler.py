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
async def manage_item_job(
    action: str,
    item: Optional[str] = None,
    item_type: Optional[str] = None,
    job_type: Optional[str] = None,
    workspace: Optional[str] = None,
    job_instance_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    schedule_action: Optional[str] = None,
    schedule_id: Optional[str] = None,
    schedule_config: Optional[Dict[str, Any]] = None,
    ctx: Context = None,
) -> str:
    """Manage on-demand jobs and schedules for Fabric items.

    Supports running, monitoring, cancelling jobs, listing job instances,
    and managing schedules for notebooks, data pipelines, Spark job definitions,
    and semantic model refreshes.

    Args:
        action: Operation to perform:
            'run' - Run an on-demand job for a Fabric item
            'get_status' - Get the status of a specific job instance
            'cancel' - Cancel a running job instance
            'list' - List recent job instances for an item
            'manage_schedule' - Create, list, update, or delete job schedules (use schedule_action)
        item: Item name or ID (required for all actions)
        item_type: Type of item: Notebook, DataPipeline, SparkJobDefinition, SemanticModel (required for all actions)
        job_type: Job type override (auto-detected from item_type if not provided)
        workspace: Workspace name or ID (uses context if not provided)
        job_instance_id: Job instance ID (required for get_status and cancel)
        parameters: Optional job parameters dict for 'run' action (passed as executionData.parameters)
        schedule_action: Sub-action for manage_schedule: 'create', 'list', 'update', 'delete'
        schedule_id: Schedule ID (required for manage_schedule update/delete)
        schedule_config: Schedule configuration dict (required for manage_schedule create/update).
            Example: {"enabled": true, "configuration": {"startDateTime": "2025-01-01T00:00:00Z", "type": "Daily", "interval": 1}}
        ctx: Context object

    Returns:
        Job or schedule details, status, or confirmation message
    """
    try:
        if not item:
            return "Error: 'item' parameter is required."
        if not item_type:
            return "Error: 'item_type' parameter is required."

        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        workspace_id, item_id = await _resolve_workspace_and_item(
            fabric_client, workspace, item, item_type, ctx.client_id
        )

        # --- run ---
        if action == "run":
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
            if response and isinstance(response, dict):
                if response.get("_status_code") == 202:
                    # Extract job instance ID from Location header URL
                    location = response.get("_location", "")
                    instance_id = location.rstrip("/").split("/")[-1] if location else "Submitted"
                    status = "Accepted"
                else:
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
            markdown += "Use `manage_item_job(action='get_status', ...)` to check progress or `manage_item_job(action='cancel', ...)` to cancel."

            return markdown

        # --- get_status ---
        elif action == "get_status":
            if not job_instance_id:
                return "Error: 'job_instance_id' is required for 'get_status' action."

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

        # --- cancel ---
        elif action == "cancel":
            if not job_instance_id:
                return "Error: 'job_instance_id' is required for 'cancel' action."

            endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}/cancel"
            await fabric_client._make_request(
                endpoint=endpoint,
                method="POST",
                params={},
            )

            return f"Job instance '{job_instance_id}' cancellation requested successfully."

        # --- list ---
        elif action == "list":
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
                jtype = job.get("jobType", "N/A")
                start_time = job.get("startTimeUtc", "N/A")
                end_time = job.get("endTimeUtc", "N/A")
                markdown += f"| {instance_id} | {status} | {jtype} | {start_time} | {end_time} |\n"

            markdown += f"\n**Total:** {len(results)} instance(s)"
            return markdown

        # --- manage_schedule ---
        elif action == "manage_schedule":
            if not schedule_action:
                return "Error: 'schedule_action' is required for 'manage_schedule' action. Use 'create', 'list', 'update', or 'delete'."

            resolved_job_type = job_type or _JOB_TYPE_DEFAULTS.get(item_type, "DefaultJob")
            base_endpoint = f"workspaces/{workspace_id}/items/{item_id}/jobs/{resolved_job_type}/schedules"

            if schedule_action == "list":
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

            elif schedule_action == "create":
                if not schedule_config:
                    return "Error: schedule_config is required for 'create' schedule_action."

                result = await fabric_client._make_request(
                    endpoint=base_endpoint,
                    method="POST",
                    params=schedule_config,
                )

                if result and isinstance(result, dict):
                    return f"Schedule created successfully. ID: {result.get('id', 'N/A')}"
                return "Schedule creation submitted."

            elif schedule_action == "update":
                if not schedule_id:
                    return "Error: schedule_id is required for 'update' schedule_action."
                if not schedule_config:
                    return "Error: schedule_config is required for 'update' schedule_action."

                endpoint = f"{base_endpoint}/{schedule_id}"
                result = await fabric_client._make_request(
                    endpoint=endpoint,
                    method="PATCH",
                    params=schedule_config,
                )

                return f"Schedule '{schedule_id}' updated successfully."

            elif schedule_action == "delete":
                if not schedule_id:
                    return "Error: schedule_id is required for 'delete' schedule_action."

                endpoint = f"{base_endpoint}/{schedule_id}"
                await fabric_client._make_request(
                    endpoint=endpoint,
                    method="DELETE",
                )

                return f"Schedule '{schedule_id}' deleted successfully."

            else:
                return f"Error: Unknown schedule_action '{schedule_action}'. Use 'create', 'list', 'update', or 'delete'."

        else:
            return f"Error: Unknown action '{action}'. Use 'run', 'get_status', 'cancel', 'list', or 'manage_schedule'."

    except Exception as e:
        return f"Error managing item job: {str(e)}"
