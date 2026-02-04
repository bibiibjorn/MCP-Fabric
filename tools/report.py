from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    ReportClient,
    PowerBIClient,
)
from helpers.logging_config import get_logger
from typing import Optional, List, Dict
import base64

logger = get_logger(__name__)


@mcp.tool()
async def list_reports(workspace: Optional[str] = None, ctx: Context = None) -> str:
    """List all reports in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace (optional)
        ctx: Context object containing client information
    Returns:
        A string containing the list of reports or an error message.
    """
    try:
        # Get workspace from parameter or cache
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        client = ReportClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        reports = await client.list_reports(ws)

        if not reports:
            return f"No reports found in workspace '{ws}'."

        markdown = f"# Reports in workspace '{ws}'\n\n"
        markdown += "| ID | Name | Description |\n"
        markdown += "|-----|------|-------------|\n"

        for report in reports:
            markdown += f"| {report.get('id', 'N/A')} | {report.get('displayName', 'N/A')} | {report.get('description', 'N/A')} |\n"

        markdown += f"\n*{len(reports)} report(s) found*"

        return markdown

    except Exception as e:
        return f"Error listing reports: {str(e)}"


@mcp.tool()
async def get_report(
    workspace: Optional[str] = None,
    report_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get a specific report by ID.

    Args:
        workspace: Name or ID of the workspace (optional)
        report_id: ID of the report (required)
        ctx: Context object containing client information

    Returns:
        A string containing the report details or an error message.
    """
    try:
        # Get workspace from parameter or cache
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        if not report_id:
            return "Error: report_id is required."

        client = ReportClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        report = await client.get_report(ws, report_id)

        if not report:
            return f"No report found with ID '{report_id}' in workspace '{ws}'."

        return f"Report details:\n\n{report}"

    except Exception as e:
        return f"Error getting report: {str(e)}"


@mcp.tool()
async def get_report_pages(
    report: str,
    workspace: Optional[str] = None,
    ctx: Context = None
) -> str:
    """List all pages in a Power BI report.

    Args:
        report: Report name or ID
        workspace: Workspace name or ID (uses context if not provided)
        ctx: Context object

    Returns:
        List of report pages with names, display names, and order

    Example:
        get_report_pages("Sales Dashboard")
        get_report_pages("abc-123-def", workspace="Finvision")
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve report
        report_id = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)

        # Get report name for display
        reports = await fabric_client.get_reports(workspace_id)
        report_info = next((r for r in reports if r.get("id") == report_id), {})
        report_name = report_info.get("displayName", report)

        # Get pages
        pages = await powerbi_client.get_report_pages(workspace_id, report_id)

        if not pages:
            return f"No pages found in report '{report_name}'"

        # Format as markdown
        markdown = f"# Pages in Report: {report_name}\n\n"
        markdown += "| Order | Name | Display Name |\n"
        markdown += "|-------|------|-------------|\n"

        for page in sorted(pages, key=lambda p: p.get("order", 0)):
            order = page.get("order", "N/A")
            name = page.get("name", "N/A")
            display_name = page.get("displayName", name)
            markdown += f"| {order} | {name} | {display_name} |\n"

        markdown += f"\n*{len(pages)} page(s) found*"

        return markdown

    except Exception as e:
        logger.error(f"Error getting report pages: {e}")
        return f"Error getting report pages: {str(e)}"


@mcp.tool()
async def export_report(
    report: str,
    workspace: Optional[str] = None,
    format: str = "PDF",
    page_name: Optional[str] = None,
    visual_name: Optional[str] = None,
    save_path: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Export a Power BI report to a file (PDF, PPTX, PNG, or other formats).

    Args:
        report: Report name or ID
        workspace: Workspace name or ID (uses context if not provided)
        format: Export format - "PDF" (default), "PPTX", "PNG", "IMAGE", "XLSX", "DOCX", "CSV", "XML", "MHTML"
        page_name: Specific page to export (optional, exports all pages if not specified)
        visual_name: Specific visual to export (optional, requires page_name)
        save_path: Local file path to save the export (optional, returns download URL if not specified)
        ctx: Context object

    Returns:
        Export status with download URL or file save confirmation

    Examples:
        export_report("Sales Dashboard", format="PDF")
        export_report("Sales Dashboard", page_name="ReportSection1", format="PNG")
        export_report("Sales Dashboard", save_path="C:/exports/report.pdf")

    Note:
        - Requires Premium, Embedded, or Fabric capacity
        - Export is asynchronous - tool will poll until complete (up to 5 minutes)
        - Large reports may take several minutes to export
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve report
        report_id = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)

        # Get report name for display
        reports = await fabric_client.get_reports(workspace_id)
        report_info = next((r for r in reports if r.get("id") == report_id), {})
        report_name = report_info.get("displayName", report)

        # Build pages configuration if specific page/visual requested
        pages_config = None
        if page_name:
            page_config = {"pageName": page_name}
            if visual_name:
                page_config["visualName"] = visual_name
            pages_config = [page_config]

        # Validate format
        valid_formats = ["PDF", "PPTX", "PNG", "IMAGE", "XLSX", "DOCX", "CSV", "XML", "MHTML"]
        format_upper = format.upper()
        if format_upper not in valid_formats:
            return f"Error: Invalid format '{format}'. Valid formats: {', '.join(valid_formats)}"

        logger.info(f"Starting export of report '{report_name}' to {format_upper}")

        # Start export job
        export_job = await powerbi_client.export_report(
            workspace_id,
            report_id,
            format=format_upper,
            pages=pages_config
        )

        export_id = export_job.get("id")
        if not export_id:
            return f"Error: Export job failed to start. Response: {export_job}"

        logger.info(f"Export job started with ID: {export_id}")

        # Wait for export to complete
        final_status = await powerbi_client.wait_for_export(
            workspace_id,
            report_id,
            export_id,
            poll_interval=5,
            timeout=300  # 5 minutes
        )

        # Get the download URL
        resource_url = final_status.get("resourceLocation", "")

        result = f"## Export Complete\n\n"
        result += f"**Report:** {report_name}\n"
        result += f"**Format:** {format_upper}\n"
        if page_name:
            result += f"**Page:** {page_name}\n"
        if visual_name:
            result += f"**Visual:** {visual_name}\n"

        # If save_path provided, download and save the file
        if save_path:
            try:
                file_content = await powerbi_client.get_export_file(workspace_id, report_id, export_id)

                with open(save_path, "wb") as f:
                    f.write(file_content)

                result += f"\n**Saved to:** {save_path}\n"
                result += f"**File size:** {len(file_content):,} bytes"

            except Exception as e:
                result += f"\n**Warning:** Could not save file: {str(e)}\n"
                result += f"**Download URL:** {resource_url}"
        else:
            result += f"\n**Download URL:** {resource_url}\n"
            result += "\n*Note: Use the download URL to retrieve the file, or provide save_path parameter to save directly.*"

        return result

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        return f"Error exporting report: {str(e)}"


@mcp.tool()
async def get_report_definition(
    report: str,
    workspace: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get the structure/definition of a Power BI report.

    Returns information about the report's internal structure including
    pages, visuals, filters, and configuration. Uses the Fabric API
    getDefinition endpoint.

    Args:
        report: Report name or ID
        workspace: Workspace name or ID (uses context if not provided)
        ctx: Context object

    Returns:
        Report definition including:
        - Pages and their layout
        - Visuals on each page
        - Filters (report, page, visual level)
        - Configuration settings

    Note:
        Requires read access to the report definition.
        Returns PBIR (Power BI Report) definition format.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve report
        report_id = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)

        # Get report name for display
        reports = await fabric_client.get_reports(workspace_id)
        report_info = next((r for r in reports if r.get("id") == report_id), {})
        report_name = report_info.get("displayName", report)

        logger.info(f"Getting definition for report '{report_name}'")

        # Call getDefinition endpoint via Fabric API
        definition = await fabric_client._make_request(
            f"workspaces/{workspace_id}/reports/{report_id}/getDefinition",
            method="POST",
            lro=True,
            lro_poll_interval=2,
            lro_timeout=120
        )

        if not definition:
            return f"Error: Could not retrieve definition for report '{report_name}'"

        # Parse the definition and format as markdown
        markdown = f"# Report Definition: {report_name}\n\n"

        # The definition contains parts with various report files
        parts = definition.get("definition", {}).get("parts", [])

        if not parts:
            return f"No definition parts found for '{report_name}'"

        # Categorize the parts
        config_parts = []
        page_parts = []
        other_parts = []

        for part in parts:
            path = part.get("path", "")
            payload = part.get("payload", "")

            if not payload:
                continue

            try:
                content = base64.b64decode(payload).decode("utf-8")
            except Exception:
                content = "[Binary content]"

            if "definition.pbir" in path or "report.json" in path:
                config_parts.append((path, content))
            elif "/pages/" in path:
                page_parts.append((path, content))
            else:
                other_parts.append((path, content))

        # Format configuration
        if config_parts:
            markdown += "## Report Configuration\n\n"
            for path, content in config_parts:
                markdown += f"### {path}\n\n"
                # Truncate if too long
                if len(content) > 3000:
                    markdown += f"```json\n{content[:3000]}...\n```\n\n"
                else:
                    markdown += f"```json\n{content}\n```\n\n"

        # Format pages
        if page_parts:
            markdown += "## Pages\n\n"
            for path, content in sorted(page_parts):
                page_name = path.split("/pages/")[-1].split("/")[0] if "/pages/" in path else path
                markdown += f"### Page: {page_name}\n\n"
                if len(content) > 2000:
                    markdown += f"```json\n{content[:2000]}...\n```\n\n"
                else:
                    markdown += f"```json\n{content}\n```\n\n"

        # Summary
        markdown += f"\n## Summary\n\n"
        markdown += f"- **Configuration files:** {len(config_parts)}\n"
        markdown += f"- **Page files:** {len(page_parts)}\n"
        markdown += f"- **Other files:** {len(other_parts)}\n"

        if other_parts:
            markdown += f"\n### Other Files\n\n"
            for path, _ in other_parts[:10]:  # List first 10 only
                markdown += f"- {path}\n"
            if len(other_parts) > 10:
                markdown += f"- ... and {len(other_parts) - 10} more\n"

        return markdown

    except Exception as e:
        logger.error(f"Error getting report definition: {e}")
        return f"Error getting report definition: {str(e)}"
