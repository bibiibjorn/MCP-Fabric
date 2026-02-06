from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import FabricApiClient
from helpers.logging_config import get_logger
from typing import Optional, Dict, List, Any

logger = get_logger(__name__)


@mcp.tool()
async def manage_deployment(
    action: str,
    pipeline_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    stage_id: Optional[str] = None,
    workspace: Optional[str] = None,
    source_stage_id: Optional[str] = None,
    target_stage_id: Optional[str] = None,
    items: Optional[List[Dict[str, str]]] = None,
    note: Optional[str] = None,
    allow_create_artifact: bool = True,
    allow_overwrite_artifact: bool = True,
    ctx: Context = None,
) -> str:
    """Manage deployment pipelines, stages, and deployments in Microsoft Fabric.

    Deployment pipelines enable CI/CD for Fabric items across
    development, test, and production stages.

    Args:
        action: Operation to perform:
            Pipeline CRUD:
                'list_pipelines' - List all deployment pipelines
                'get_pipeline' - Get details of a specific pipeline
                'create_pipeline' - Create a new deployment pipeline
                'update_pipeline' - Update a pipeline's name/description
                'delete_pipeline' - Delete a deployment pipeline
            Stage management:
                'list_stages' - List all stages in a pipeline
                'assign_workspace' - Assign a workspace to a stage
                'unassign_workspace' - Remove workspace from a stage
                'list_stage_items' - List items deployed in a stage
            Deploy:
                'deploy' - Deploy content from one stage to another
        pipeline_id: Pipeline ID (required for most actions except list_pipelines and create_pipeline)
        name: Pipeline display name (required for create_pipeline, optional for update_pipeline)
        description: Pipeline description (optional for create_pipeline/update_pipeline)
        stage_id: Stage ID (required for assign_workspace, unassign_workspace, list_stage_items)
        workspace: Workspace name or ID (required for assign_workspace)
        source_stage_id: Source stage ID for deploy (required for deploy)
        target_stage_id: Target stage ID for deploy (optional, defaults to next stage)
        items: Specific items to deploy (optional for deploy).
            Each item: {"sourceItemId": "...", "targetItemId": "..."} or {"sourceItemId": "..."}
        note: Deployment note (optional for deploy)
        allow_create_artifact: Allow creating new items in target during deploy (default: True)
        allow_overwrite_artifact: Allow overwriting existing items during deploy (default: True)
        ctx: Context object

    Returns:
        Pipeline/stage details, item lists, deployment status, or confirmation message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        # --- list_pipelines ---
        if action == "list_pipelines":
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

        # --- get_pipeline ---
        elif action == "get_pipeline":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'get_pipeline' action."

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

        # --- create_pipeline ---
        elif action == "create_pipeline":
            if not name:
                return "Error: name is required for 'create_pipeline' action."

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

        # --- update_pipeline ---
        elif action == "update_pipeline":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'update_pipeline' action."

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

        # --- delete_pipeline ---
        elif action == "delete_pipeline":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'delete_pipeline' action."

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}",
                method="DELETE",
            )

            return f"Deployment pipeline '{pipeline_id}' deleted successfully."

        # --- list_stages ---
        elif action == "list_stages":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'list_stages' action."

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

        # --- assign_workspace ---
        elif action == "assign_workspace":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'assign_workspace' action."
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

        # --- unassign_workspace ---
        elif action == "unassign_workspace":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'unassign_workspace' action."
            if not stage_id:
                return "Error: stage_id is required for 'unassign_workspace' action."

            await fabric_client._make_request(
                endpoint=f"deploymentPipelines/{pipeline_id}/stages/{stage_id}/unassignWorkspace",
                method="POST",
                params={},
            )

            return f"Workspace unassigned from stage '{stage_id}' successfully."

        # --- list_stage_items ---
        elif action == "list_stage_items":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'list_stage_items' action."
            if not stage_id:
                return "Error: stage_id is required for 'list_stage_items' action."

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

        # --- deploy ---
        elif action == "deploy":
            if not pipeline_id:
                return "Error: pipeline_id is required for 'deploy' action."
            if not source_stage_id:
                return "Error: source_stage_id is required for 'deploy' action."

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

        else:
            return f"Error: Unknown action '{action}'. Use 'list_pipelines', 'get_pipeline', 'create_pipeline', 'update_pipeline', 'delete_pipeline', 'list_stages', 'assign_workspace', 'unassign_workspace', 'list_stage_items', or 'deploy'."

    except Exception as e:
        return f"Error managing deployment: {str(e)}"
