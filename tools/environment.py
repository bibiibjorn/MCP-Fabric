from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import FabricApiClient
from helpers.logging_config import get_logger
from typing import Optional, Dict, Any

logger = get_logger(__name__)


async def _resolve_workspace_for_env(
    fabric_client: FabricApiClient,
    workspace: Optional[str],
    client_id: str,
) -> str:
    """Resolve workspace ID from parameter or context."""
    ws = workspace or __ctx_cache.get(f"{client_id}_workspace")
    if not ws:
        raise ValueError("Workspace not set. Use set_context('workspace', ...) or provide workspace parameter.")
    return await fabric_client.resolve_workspace(ws)


@mcp.tool()
async def manage_environment(
    action: str,
    workspace: Optional[str] = None,
    environment_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    compute_config: Optional[Dict[str, Any]] = None,
    library_name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage Fabric Spark environments, their compute configuration, and libraries.

    Environments provide Spark compute and library configurations
    for notebooks and Spark job definitions.

    Args:
        action: Operation to perform:
            Environment CRUD:
                'list' - List all environments in a workspace
                'get' - Get details of a specific environment
                'create' - Create a new environment
                'update' - Update an environment's name/description
                'delete' - Delete an environment
            Compute configuration:
                'get_compute' - Get Spark compute configuration
                'update_compute' - Update Spark compute configuration (staged)
            Library management:
                'list_libraries' - List libraries in an environment
                'delete_library' - Delete a library from an environment (staged)
            Publish:
                'publish' - Publish all staged changes (compute/library)
        workspace: Workspace name or ID (uses context if not provided)
        environment_id: Environment ID (required for get, update, delete, compute, library, and publish actions)
        name: Environment display name (required for create, optional for update)
        description: Description (optional for create/update)
        compute_config: Spark compute configuration (required for update_compute).
            Example: {
                "instancePool": {"name": "starter", "type": "Workspace"},
                "driverCores": 4, "driverMemory": "28g",
                "executorCores": 4, "executorMemory": "28g",
                "dynamicExecutorAllocation": {"enabled": true, "minExecutors": 1, "maxExecutors": 2}
            }
        library_name: Library name to delete (required for delete_library)
        ctx: Context object

    Returns:
        Environment details, compute configuration, library list, or confirmation message
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        workspace_id = await _resolve_workspace_for_env(fabric_client, workspace, ctx.client_id)

        base_endpoint = f"workspaces/{workspace_id}/environments"

        # --- list ---
        if action == "list":
            results = await fabric_client._make_request(
                endpoint=base_endpoint,
                use_pagination=True,
            )

            if not results:
                return "No environments found in this workspace."

            markdown = "# Environments\n\n"
            markdown += "| ID | Name | Description |\n"
            markdown += "|---|---|---|\n"

            for env in results:
                eid = env.get("id", "N/A")
                ename = env.get("displayName", "N/A")
                desc = env.get("description", "")
                markdown += f"| {eid} | {ename} | {desc} |\n"

            markdown += f"\n**Total:** {len(results)} environment(s)"
            return markdown

        # --- get ---
        elif action == "get":
            if not environment_id:
                return "Error: environment_id is required for 'get' action."

            result = await fabric_client._make_request(
                endpoint=f"{base_endpoint}/{environment_id}",
            )

            if not result:
                return f"Error: Could not retrieve environment '{environment_id}'."

            markdown = "# Environment Details\n\n"
            markdown += f"**ID:** {result.get('id', 'N/A')}\n"
            markdown += f"**Name:** {result.get('displayName', 'N/A')}\n"
            markdown += f"**Description:** {result.get('description', 'N/A')}\n"
            markdown += f"**Type:** {result.get('type', 'N/A')}\n"

            # Show properties if available
            props = result.get("properties", {})
            if props:
                publish_details = props.get("publishDetails", {})
                state = publish_details.get("state", "N/A")
                target_version = publish_details.get("targetVersion", "N/A")
                markdown += f"**Publish State:** {state}\n"
                markdown += f"**Target Version:** {target_version}\n"

            return markdown

        # --- create ---
        elif action == "create":
            if not name:
                return "Error: name is required for 'create' action."

            body = {"displayName": name}
            if description:
                body["description"] = description

            result = await fabric_client._make_request(
                endpoint=base_endpoint,
                method="POST",
                params=body,
            )

            if result and isinstance(result, dict):
                eid = result.get("id", "N/A")
                return f"Environment '{name}' created successfully.\n\n**ID:** {eid}"
            return "Environment creation submitted."

        # --- update ---
        elif action == "update":
            if not environment_id:
                return "Error: environment_id is required for 'update' action."

            body = {}
            if name:
                body["displayName"] = name
            if description is not None:
                body["description"] = description

            if not body:
                return "Error: Provide name and/or description to update."

            await fabric_client._make_request(
                endpoint=f"{base_endpoint}/{environment_id}",
                method="PATCH",
                params=body,
            )

            return f"Environment '{environment_id}' updated successfully."

        # --- delete ---
        elif action == "delete":
            if not environment_id:
                return "Error: environment_id is required for 'delete' action."

            await fabric_client._make_request(
                endpoint=f"{base_endpoint}/{environment_id}",
                method="DELETE",
            )

            return f"Environment '{environment_id}' deleted successfully."

        # --- get_compute ---
        elif action == "get_compute":
            if not environment_id:
                return "Error: environment_id is required for 'get_compute' action."

            endpoint = f"workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute"
            result = await fabric_client._make_request(endpoint=endpoint)

            if not result:
                return f"Error: Could not retrieve compute config for environment '{environment_id}'."

            markdown = "# Spark Compute Configuration\n\n"
            markdown += f"**Environment ID:** {environment_id}\n\n"

            # Instance pool
            pool = result.get("instancePool", {})
            if pool:
                markdown += f"**Instance Pool:** {pool.get('name', 'N/A')} ({pool.get('type', 'N/A')})\n"

            # Driver config
            markdown += f"**Driver Cores:** {result.get('driverCores', 'N/A')}\n"
            markdown += f"**Driver Memory:** {result.get('driverMemory', 'N/A')}\n"

            # Executor config
            markdown += f"**Executor Cores:** {result.get('executorCores', 'N/A')}\n"
            markdown += f"**Executor Memory:** {result.get('executorMemory', 'N/A')}\n"

            # Dynamic allocation
            dynamic = result.get("dynamicExecutorAllocation", {})
            if dynamic:
                markdown += f"**Dynamic Allocation:** {'Enabled' if dynamic.get('enabled') else 'Disabled'}\n"
                markdown += f"**Min Executors:** {dynamic.get('minExecutors', 'N/A')}\n"
                markdown += f"**Max Executors:** {dynamic.get('maxExecutors', 'N/A')}\n"

            # Runtime version
            runtime = result.get("runtimeVersion", "N/A")
            markdown += f"**Runtime Version:** {runtime}\n"

            markdown += "\n*Changes are staged. Use manage_environment(action='publish', ...) to apply.*"
            return markdown

        # --- update_compute ---
        elif action == "update_compute":
            if not environment_id:
                return "Error: environment_id is required for 'update_compute' action."
            if not compute_config:
                return "Error: compute_config is required for 'update_compute' action."

            endpoint = f"workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute"
            await fabric_client._make_request(
                endpoint=endpoint,
                method="PATCH",
                params=compute_config,
            )

            return f"Spark compute configuration updated (staged) for environment '{environment_id}'.\n\nUse `manage_environment(action='publish', ...)` to apply changes."

        # --- list_libraries ---
        elif action == "list_libraries":
            if not environment_id:
                return "Error: environment_id is required for 'list_libraries' action."

            endpoint = f"workspaces/{workspace_id}/environments/{environment_id}/staging/libraries"
            result = await fabric_client._make_request(endpoint=endpoint)

            if not result:
                return f"No libraries found for environment '{environment_id}'."

            markdown = "# Environment Libraries\n\n"
            markdown += f"**Environment ID:** {environment_id}\n\n"

            # Custom libraries
            custom_libs = result.get("customLibraries", {})
            if custom_libs:
                markdown += "## Custom Libraries\n\n"
                for lib_type, libs in custom_libs.items():
                    if libs:
                        markdown += f"### {lib_type}\n"
                        for lib in libs:
                            if isinstance(lib, str):
                                markdown += f"- {lib}\n"
                            elif isinstance(lib, dict):
                                markdown += f"- {lib.get('name', 'N/A')}\n"

            # Environment YAML / pip dependencies
            env_yml = result.get("environmentYml", None)
            if env_yml:
                markdown += "\n## Environment YAML\n\n"
                markdown += f"```yaml\n{env_yml}\n```\n"

            markdown += "\n*Changes are staged. Use manage_environment(action='publish', ...) to apply.*"
            return markdown

        # --- delete_library ---
        elif action == "delete_library":
            if not environment_id:
                return "Error: environment_id is required for 'delete_library' action."
            if not library_name:
                return "Error: library_name is required for 'delete_library' action."

            endpoint = f"workspaces/{workspace_id}/environments/{environment_id}/staging/libraries"
            await fabric_client._make_request(
                endpoint=endpoint,
                method="DELETE",
                params={"libraryToDelete": library_name},
            )

            return f"Library '{library_name}' removed (staged) from environment '{environment_id}'.\n\nUse `manage_environment(action='publish', ...)` to apply changes."

        # --- publish ---
        elif action == "publish":
            if not environment_id:
                return "Error: environment_id is required for 'publish' action."

            endpoint = f"workspaces/{workspace_id}/environments/{environment_id}/staging/publish"

            result = await fabric_client._make_request(
                endpoint=endpoint,
                method="POST",
                params={},
                lro=True,
                lro_poll_interval=5,
                lro_timeout=600,
            )

            if result and isinstance(result, dict):
                status = result.get("status", result.get("operationStatus", "Unknown"))
                markdown = "# Environment Publish Result\n\n"
                markdown += f"**Environment ID:** {environment_id}\n"
                markdown += f"**Status:** {status}\n"

                if status in ("Succeeded", "succeeded", "Completed", "completed"):
                    markdown += "\nAll staged changes have been published successfully."
                elif status in ("Failed", "failed"):
                    error = result.get("error", {})
                    markdown += f"\n**Error:** {error.get('message', 'Unknown error')}\n"

                return markdown

            return "Environment publish operation submitted. Changes may take a few minutes to apply."

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'get', 'create', 'update', 'delete', 'get_compute', 'update_compute', 'list_libraries', 'delete_library', or 'publish'."

    except Exception as e:
        return f"Error managing environment: {str(e)}"
