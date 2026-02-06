from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    LakehouseClient,
)
from helpers.logging_config import get_logger

# import sempy_labs as labs
# import sempy_labs.lakehouse as slh

from typing import Optional

logger = get_logger(__name__)


@mcp.tool()
async def list_lakehouses(workspace: Optional[str] = None, ctx: Context = None) -> str:
    """List all lakehouses in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace (optional)
        ctx: Context object containing client information

    Returns:
        A string containing the list of lakehouses or an error message.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential=credential)
        lakehouse_client = LakehouseClient(client=fabric_client)
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using the 'set_workspace' command."
        return await lakehouse_client.list_lakehouses(workspace=ws)
    except Exception as e:
        logger.error(f"Error listing lakehouses: {e}")
        return f"Error listing lakehouses: {e}"


# @mcp.tool()
# async def list_lakehouses_semantic_link(workspace: Optional[str] = None, ctx: Context = None) -> str:
#     """List all lakehouses in a Fabric workspace using semantic-link-labs."""
#     try:
#         manager = LakehouseManager()
#         lakehouses = manager.list_lakehouses(workspace_id=workspace or __ctx_cache.get(f"{ctx.client_id}_workspace"))
#         markdown = f"# Lakehouses (semantic-link-labs) in workspace '{workspace}'\n\n"
#         markdown += "| ID | Name |\n"
#         markdown += "|-----|------|\n"
#         for lh in lakehouses:
#             markdown += f"| {lh.get('id', 'N/A')} | {lh.get('displayName', 'N/A')} |\n"
#         return markdown
#     except Exception as e:
#         return f"Error listing lakehouses with semantic-link-labs: {str(e)}"


@mcp.tool()
async def create_lakehouse(
    name: str,
    workspace: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Create a new lakehouse in a Fabric workspace.

    Args:
        name: Name of the lakehouse
        workspace: Name or ID of the workspace (optional)
        description: Description of the lakehouse (optional)
        ctx: Context object containing client information
    Returns:
        A string confirming the lakehouse has been created or an error message.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential=credential)
        lakehouse_client = LakehouseClient(client=fabric_client)
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using the 'set_workspace' command."
        return await lakehouse_client.create_lakehouse(
            name=name, workspace=ws, description=description
        )
    except Exception as e:
        logger.error(f"Error creating lakehouse: {e}")
        return f"Error creating lakehouse: {e}"


@mcp.tool()
async def list_shortcuts(
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    ctx: Context = None
) -> str:
    """List all shortcuts in a lakehouse.

    Shortcuts provide virtualized access to external data sources without copying data.
    They can point to other OneLake locations, Azure Data Lake Storage, Amazon S3, or
    Google Cloud Storage.

    Args:
        workspace: Workspace name or ID (optional - uses context if not provided)
        lakehouse: Lakehouse name or ID (optional - uses context if not provided)
        ctx: Context object containing client information

    Returns:
        List of shortcuts with name, path, target location, and type
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential=credential)
        lakehouse_client = LakehouseClient(client=fabric_client)

        # Resolve workspace
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using 'set_workspace' command."

        workspace_id = await fabric_client.resolve_workspace(ws)

        # Resolve lakehouse
        lh = lakehouse or __ctx_cache.get(f"{ctx.client_id}_lakehouse")
        if not lh:
            return "Lakehouse not set. Please set a lakehouse using 'set_lakehouse' command or provide lakehouse parameter."

        lakehouse_id = await fabric_client.resolve_item_id(
            item=lh, type="Lakehouse", workspace=workspace_id
        )

        # Get shortcuts
        shortcuts = await lakehouse_client.list_shortcuts(workspace_id, lakehouse_id)

        if not shortcuts:
            return f"No shortcuts found in lakehouse '{lh}' in workspace '{ws}'."

        # Format as markdown table
        markdown = f"# Shortcuts in lakehouse '{lh}'\n\n"
        markdown += "Shortcuts provide virtualized access to external data without physical copying.\n\n"
        markdown += "| Name | Path | Target Location | Connection Type |\n"
        markdown += "|------|------|-----------------|----------------|\n"

        for shortcut in shortcuts:
            name = shortcut.get("name", "N/A")
            path = shortcut.get("path", "N/A")

            # Extract target information
            target = shortcut.get("target", {})
            if isinstance(target, dict):
                # OneLake shortcut
                if "oneLake" in target:
                    onelake_info = target["oneLake"]
                    workspace_name = onelake_info.get("workspaceName", "N/A")
                    item_name = onelake_info.get("itemName", "N/A")
                    target_path = onelake_info.get("path", "")
                    target_location = f"OneLake: {workspace_name}/{item_name}{target_path}"
                    connection_type = "OneLake"

                # ADLS Gen2 shortcut
                elif "adlsGen2" in target:
                    adls_info = target["adlsGen2"]
                    endpoint = adls_info.get("endpoint", "N/A")
                    location = adls_info.get("location", "")
                    target_location = f"{endpoint}{location}"
                    connection_type = "ADLS Gen2"

                # S3 shortcut
                elif "s3" in target:
                    s3_info = target["s3"]
                    endpoint = s3_info.get("endpoint", "N/A")
                    location = s3_info.get("location", "")
                    target_location = f"s3://{endpoint}{location}"
                    connection_type = "Amazon S3"

                # Google Cloud Storage shortcut
                elif "googleCloudStorage" in target:
                    gcs_info = target["googleCloudStorage"]
                    endpoint = gcs_info.get("endpoint", "N/A")
                    location = gcs_info.get("location", "")
                    target_location = f"gs://{endpoint}{location}"
                    connection_type = "GCS"

                else:
                    target_location = str(target)
                    connection_type = "Unknown"
            else:
                target_location = str(target)
                connection_type = "Unknown"

            markdown += f"| {name} | {path} | {target_location} | {connection_type} |\n"

        markdown += f"\n**Total shortcuts:** {len(shortcuts)}\n"
        markdown += "\n### Shortcut Types:\n"
        markdown += "- **OneLake**: Access data from other Fabric lakehouses\n"
        markdown += "- **ADLS Gen2**: Access Azure Data Lake Storage\n"
        markdown += "- **Amazon S3**: Access AWS S3 buckets\n"
        markdown += "- **GCS**: Access Google Cloud Storage\n"

        return markdown

    except Exception as e:
        logger.error(f"Error listing shortcuts: {str(e)}")
        return f"Error listing shortcuts: {str(e)}"


# ===== AGENTIC / ORCHESTRATED TOOLS =====
# These tools use the orchestration layer for intelligent multi-step workflows

@mcp.tool()
async def explore_lakehouse_complete(
    workspace: str,
    lakehouse: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Comprehensive lakehouse exploration workflow (AGENTIC).
    
    This is an intelligent orchestrated operation that:
    1. Lists all lakehouses (if lakehouse not specified)
    2. Lists all tables in the lakehouse
    3. Gets schemas for all tables
    4. Generates sample code for data access
    5. Provides intelligent next-step suggestions
    
    Args:
        workspace: Fabric workspace name
        lakehouse: Specific lakehouse name (optional, explores all if not provided)
        ctx: Context object
    
    Returns:
        JSON string with complete exploration results and suggestions
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.lakehouse_orchestrator import lakehouse_orchestrator
        import json
        
        result = await lakehouse_orchestrator.explore_lakehouse_complete(
            workspace=workspace,
            lakehouse=lakehouse,
            ctx=ctx
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in explore_lakehouse_complete: {str(e)}")
        import json
        return json.dumps({
            "error": str(e),
            "success": False
        }, indent=2)


@mcp.tool()
async def execute_lakehouse_intent(
    goal: str,
    workspace: Optional[str] = None,
    lakehouse: Optional[str] = None,
    execution_mode: str = "standard",
    ctx: Context = None
) -> str:
    """Execute lakehouse operations based on natural language intent (AGENTIC).
    
    Intelligent routing based on keywords in the goal:
    - Explore/discover → comprehensive lakehouse exploration
    - Integrate/connect → data integration setup
    - Analyze/performance → lakehouse performance analysis
    - Read/load → data reading workflow
    - Write/save → data writing workflow
    
    Execution modes:
    - fast: Quick preview, minimal validation (5x faster)
    - standard: Normal execution with validation (default)
    - analyze: Full analysis with performance metrics
    - safe: Maximum validation with rollback capability
    
    Args:
        goal: Natural language description of what you want to achieve
        workspace: Fabric workspace name (optional, uses context if not provided)
        lakehouse: Lakehouse name (optional, uses context if not provided)
        execution_mode: Execution mode (fast/standard/analyze/safe)
        ctx: Context object
    
    Returns:
        JSON string with execution results and intelligent suggestions
    
    Examples:
        - "Explore all tables in the lakehouse"
        - "Setup data integration to read from sales table"
        - "Analyze lakehouse performance and structure"
        - "Discover what data is available"
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.agent_policy import agent_policy
        from helpers.utils.context import get_context
        import json
        
        # Get context
        ctx_obj = get_context()
        context = {
            'workspace': workspace or ctx_obj.workspace,
            'lakehouse': lakehouse or ctx_obj.lakehouse,
        }
        
        # Execute intent
        result = await agent_policy.execute_intent(
            intent=goal,
            domain='lakehouse',
            context=context,
            execution_mode=execution_mode,
            ctx=ctx
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in execute_lakehouse_intent: {str(e)}")
        import json
        return json.dumps({
            "error": str(e),
            "success": False,
            "suggestion": "Try rephrasing your goal or provide more specific details"
        }, indent=2)


@mcp.tool()
async def setup_data_integration(
    workspace: str,
    lakehouse: str,
    goal: str,
    ctx: Context = None
) -> str:
    """Setup data integration workflow for lakehouse (AGENTIC).
    
    Creates a complete data integration setup including:
    - Notebook creation
    - Code generation for read/write operations
    - Data quality checks
    - Intelligent suggestions for next steps
    
    Args:
        workspace: Workspace name
        lakehouse: Lakehouse name
        goal: Integration goal (e.g., "read from sales table", "write to warehouse")
        ctx: Context object
    
    Returns:
        JSON string with integration setup results
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.lakehouse_orchestrator import lakehouse_orchestrator
        import json
        
        result = await lakehouse_orchestrator.setup_data_integration(
            workspace=workspace,
            lakehouse=lakehouse,
            goal=goal,
            ctx=ctx
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in setup_data_integration: {str(e)}")
        import json
        return json.dumps({
            "error": str(e),
            "success": False
        }, indent=2)


