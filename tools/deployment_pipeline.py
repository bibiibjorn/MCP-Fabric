from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import FabricApiClient
from helpers.logging_config import get_logger
from typing import Optional, Dict, List, Any

logger = get_logger(__name__)


@mcp.tool()
async def manage_deployment_pipeline(
    action: str,
    pipeline_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Create, list, get, update, or delete deployment pipelines.

    Deployment pipelines enable CI/CD for Fabric items across
    development, test, and production stages.

    Args:
        action: Operation: 'list', 'get', 'create', 'update', 'delete'
        pipeline_id: Pipeline ID (required for get/update/delete)
        name: Pipeline display name (required for create, optional for update)
        description: Pipeline description (optional for create/update)
        ctx: Context object

    Returns:
        Pipeline details or confirmation message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        if action == "list":
            results = await fabric_client._make_request(
                endpoint="deploymentPipelines",
                use_pagination=True,
            )

            if not results:
                return "No deployment pipelines found."

            markdown = "# Deployment Pipelines\n\n"
            markdown += "| ID | Name | Description |\n"
            markdown += "|---|---|---|\n"

            for pipeline in results:
                pid = pipeline.get("id", "N/A")
                pname = pipeline.get("displayName", "N/A")
                desc = pipeline.get("description", "")
                markdown += f"| {pid} | {pname} | {desc} |\n"

            markdown += f"\n**Total:** {len(results)} pipeline(s)"
            return markdown

        elif action == "get":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'get' action."

            result = await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}",
            )

            if not result:
                return f"Error: Could not retrieve deployment pipeline '{pipeline_id}'."

            markdown = "# Deployment Pipeline Details\n\n"
            markdown += f"**ID:** {result.get('id', 'N/A')}\n"
            markdown += f"**Name:** {result.get('displayName', 'N/A')}\n"
            markdown += f"**Description:** {result.get('description', 'N/A')}\n"

            return markdown

        elif action == "create":
            if not name:
                return "Error: name is required for 'create' action."

            body = {"displayName": name}
            if description:
                body["description"] = description

            result = await fabric_client._make_request(
                endpoint="deploymentPipelines",
                method="POST",
                params=body,
            )

            if result and isinstance(result, dict):
                pid = result.get("id", "N/A")
                __ctx_cache[f"{ctx.client_id}_deployment_pipeline"] = pid
                return f"Deployment pipeline '{name}' created successfully.\n\n**ID:** {pid}"
            return "Deployment pipeline creation submitted."

        elif action == "update":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'update' action."

            body = {}
            if name:
                body["displayName"] = name
            if description is not None:
                body["description"] = description

            if not body:
                return "Error: Provide name and/or description to update."

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}",
                method="PATCH",
                params=body,
            )

            return f"Deployment pipeline '{pipeline_id}' updated successfully."

        elif action == "delete":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'delete' action."

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}",
                method="DELETE",
            )

            return f"Deployment pipeline '{pipeline_id}' deleted successfully."

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'get', 'create', 'update', or 'delete'."

    except Exception as e:
        return f"Error managing deployment pipeline: {str(e)}"


@mcp.tool()
async def manage_deployment_stages(
    action: str,
    pipeline_id: str,
    stage_id: Optional[str] = None,
    workspace: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage deployment pipeline stages.

    List stages, assign/unassign workspaces, and list items in stages.

    Args:
        action: Operation:
            'list_stages' - List all stages in the pipeline
            'get_stage' - Get a specific stage
            'assign_workspace' - Assign a workspace to a stage
            'unassign_workspace' - Remove workspace from a stage
            'list_items' - List items deployed in a stage
        pipeline_id: Deployment pipeline ID
        stage_id: Stage ID (required for get_stage, assign_workspace, unassign_workspace, list_items)
        workspace: Workspace name or ID (required for assign_workspace)
        ctx: Context object

    Returns:
        Stage details, items list, or confirmation message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        if action == "list_stages":
            results = await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages",
                use_pagination=True,
            )

            if not results:
                return f"No stages found for pipeline '{pipeline_id}'."

            markdown = "# Pipeline Stages\n\n"
            markdown += "| Order | Stage ID | Display Name | Workspace ID | Workspace Name |\n"
            markdown += "|---|---|---|---|---|\n"

            for stage in results:
                order = stage.get("order", "N/A")
                sid = stage.get("id", "N/A")
                sname = stage.get("displayName", "N/A")
                ws_id = stage.get("workspaceId", "Not assigned")
                ws_name = stage.get("workspaceName", "N/A")
                markdown += f"| {order} | {sid} | {sname} | {ws_id} | {ws_name} |\n"

            return markdown

        elif action == "get_stage":
            if not stage_id:
                return "Error: stage_id is required for 'get_stage' action."

            result = await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages/{stage_id}",
            )

            if not result:
                return f"Error: Could not retrieve stage '{stage_id}'."

            markdown = "# Stage Details\n\n"
            markdown += f"**Stage ID:** {result.get('id', 'N/A')}\n"
            markdown += f"**Display Name:** {result.get('displayName', 'N/A')}\n"
            markdown += f"**Order:** {result.get('order', 'N/A')}\n"
            markdown += f"**Workspace ID:** {result.get('workspaceId', 'Not assigned')}\n"
            markdown += f"**Workspace Name:** {result.get('workspaceName', 'N/A')}\n"

            return markdown

        elif action == "assign_workspace":
            if not stage_id:
                return "Error: stage_id is required for 'assign_workspace' action."
            if not workspace:
                return "Error: workspace is required for 'assign_workspace' action."

            workspace_id = await fabric_client.resolve_workspace(workspace)

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages/{stage_id}/assignWorkspace",
                method="POST",
                params={"workspaceId": workspace_id},
            )

            return f"Workspace '{workspace}' assigned to stage '{stage_id}' successfully."

        elif action == "unassign_workspace":
            if not stage_id:
                return "Error: stage_id is required for 'unassign_workspace' action."

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages/{stage_id}/unassignWorkspace",
                method="POST",
                params={},
            )

            return f"Workspace unassigned from stage '{stage_id}' successfully."

        elif action == "list_items":
            if not stage_id:
                return "Error: stage_id is required for 'list_items' action."

            results = await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages/{stage_id}/items",
                use_pagination=True,
            )

            if not results:
                return f"No items found in stage '{stage_id}'."

            markdown = f"# Items in Stage\n\n"
            markdown += "| Item ID | Name | Item Type | Source Item ID |\n"
            markdown += "|---|---|---|---|\n"

            for item in results:
                iid = item.get("itemId", "N/A")
                iname = item.get("itemDisplayName", "N/A")
                itype = item.get("itemType", "N/A")
                source_id = item.get("sourceItemId", "N/A")
                markdown += f"| {iid} | {iname} | {itype} | {source_id} |\n"

            markdown += f"\n**Total:** {len(results)} item(s)"
            return markdown

        else:
            return f"Error: Unknown action '{action}'. Use 'list_stages', 'get_stage', 'assign_workspace', 'unassign_workspace', or 'list_items'."

    except Exception as e:
        return f"Error managing deployment stages: {str(e)}"


@mcp.tool()
async def deploy_stage_content(
    pipeline_id: str,
    source_stage_id: str,
    target_stage_id: Optional[str] = None,
    items: Optional[List[Dict[str, str]]] = None,
    note: Optional[str] = None,
    allow_create_artifact: bool = True,
    allow_overwrite_artifact: bool = True,
    ctx: Context = None,
) -> str:
    """Deploy content from one pipeline stage to another.

    This is a long-running operation that deploys items between stages
    (e.g., from Development to Test, or Test to Production).

    Args:
        pipeline_id: Deployment pipeline ID
        source_stage_id: Source stage ID (deploy FROM this stage)
        target_stage_id: Target stage ID (deploy TO). If omitted, deploys to the next stage.
        items: Optional list of specific items to deploy.
            Each item: {"sourceItemId": "...", "targetItemId": "..."} or {"sourceItemId": "..."}
            If omitted, deploys all items.
        note: Optional deployment note
        allow_create_artifact: Allow creating new items in target (default: True)
        allow_overwrite_artifact: Allow overwriting existing items (default: True)
        ctx: Context object

    Returns:
        Deployment operation status and details
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        body = {
            "sourceStageId": source_stage_id,
            "options": {
                "allowCreateArtifact": allow_create_artifact,
                "allowOverwriteArtifact": allow_overwrite_artifact,
            },
        }

        if target_stage_id:
            body["targetStageId"] = target_stage_id
        if items:
            body["items"] = items
        if note:
            body["note"] = note

        result = await fabric_client._make_request(
            endpoint=f"deploymentPipelines/{pipeline_id}/deploy",
            method="POST",
            params=body,
            lro=True,
            lro_poll_interval=5,
            lro_timeout=600,
        )

        if result and isinstance(result, dict):
            status = result.get("status", result.get("operationStatus", "Unknown"))
            markdown = "# Deployment Result\n\n"
            markdown += f"**Pipeline ID:** {pipeline_id}\n"
            markdown += f"**Source Stage:** {source_stage_id}\n"
            markdown += f"**Target Stage:** {target_stage_id or 'Next stage'}\n"
            markdown += f"**Status:** {status}\n"

            if note:
                markdown += f"**Note:** {note}\n"

            # Check for deployment details
            if "deployedItems" in result:
                deployed = result["deployedItems"]
                markdown += f"\n**Items Deployed:** {len(deployed)}\n"

            if status in ("Succeeded", "succeeded", "Completed", "completed"):
                markdown += "\nDeployment completed successfully."
            elif status in ("Failed", "failed"):
                error = result.get("error", {})
                markdown += f"\n**Error:** {error.get('message', 'Unknown error')}\n"

            return markdown

        return "Deployment submitted. Use deployment pipeline operations to check status."

    except Exception as e:
        return f"Error deploying stage content: {str(e)}"
