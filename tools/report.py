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
import json

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


# Maximum response size to avoid "Tool result is too large" errors
MAX_RESPONSE_SIZE = 80000  # ~80KB limit for MCP tool responses


async def _get_report_definition_parsed(
    fabric_client: FabricApiClient,
    powerbi_client: PowerBIClient,
    workspace_id: str,
    report: str
) -> Dict:
    """Helper to fetch and parse report definition.

    Returns dict with:
        - report_id: str
        - report_name: str
        - config_parts: list of (path, content) tuples
        - page_parts: list of (path, content) tuples
        - other_parts: list of (path, content) tuples
        - page_names: list of page identifiers
    """
    # Resolve report
    report_id = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)

    # Get report name
    reports = await fabric_client.get_reports(workspace_id)
    report_info = next((r for r in reports if r.get("id") == report_id), {})
    report_name = report_info.get("displayName", report)

    # Get definition
    definition = await fabric_client._make_request(
        f"workspaces/{workspace_id}/reports/{report_id}/getDefinition",
        method="POST",
        lro=True,
        lro_poll_interval=2,
        lro_timeout=120
    )

    if not definition:
        raise ValueError(f"Could not retrieve definition for report '{report_name}'")

    parts = definition.get("definition", {}).get("parts", [])

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

    # Extract unique page names
    page_names = []
    for path, _ in page_parts:
        if "/pages/" in path:
            extracted = path.split("/pages/")[-1].split("/")[0]
            if extracted not in page_names:
                page_names.append(extracted)

    return {
        "report_id": report_id,
        "report_name": report_name,
        "config_parts": config_parts,
        "page_parts": page_parts,
        "other_parts": other_parts,
        "page_names": page_names
    }


def _parse_visual_from_json(content: str) -> List[Dict]:
    """Extract visual information from page JSON content."""
    visuals = []

    try:
        data = json.loads(content)

        # PBIR format has visuals in different structures
        # Check for visualContainers (common in PBIR)
        if "visualContainers" in data:
            for vc in data.get("visualContainers", []):
                visual = {
                    "name": vc.get("name", "Unknown"),
                    "type": "Unknown",
                    "x": vc.get("position", {}).get("x", 0),
                    "y": vc.get("position", {}).get("y", 0),
                    "width": vc.get("position", {}).get("width", 0),
                    "height": vc.get("position", {}).get("height", 0),
                }

                # Try to get visual type from config
                config = vc.get("config", {})
                if isinstance(config, str):
                    try:
                        config = json.loads(config)
                    except:
                        pass

                if isinstance(config, dict):
                    visual["type"] = config.get("singleVisual", {}).get("visualType", "Unknown")

                    # Get title if available
                    title_config = config.get("singleVisual", {}).get("vcObjects", {}).get("title", [])
                    if title_config and isinstance(title_config, list) and len(title_config) > 0:
                        title_props = title_config[0].get("properties", {})
                        title_text = title_props.get("text", {}).get("expr", {}).get("Literal", {}).get("Value", "")
                        if title_text:
                            visual["title"] = title_text.strip("'\"")

                visuals.append(visual)

        # Also check for 'visuals' array (alternative format)
        elif "visuals" in data:
            for v in data.get("visuals", []):
                visual = {
                    "name": v.get("name", v.get("visualName", "Unknown")),
                    "type": v.get("type", v.get("visualType", "Unknown")),
                    "title": v.get("title", ""),
                }
                if "position" in v:
                    visual.update({
                        "x": v["position"].get("x", 0),
                        "y": v["position"].get("y", 0),
                        "width": v["position"].get("width", 0),
                        "height": v["position"].get("height", 0),
                    })
                visuals.append(visual)

    except json.JSONDecodeError:
        pass

    return visuals


def _parse_filters_from_json(content: str, source: str = "report") -> List[Dict]:
    """Extract filter information from JSON content."""
    filters = []

    try:
        data = json.loads(content)

        # Check various filter locations in PBIR format
        filter_locations = [
            ("filters", data.get("filters", [])),
            ("filterConfig", data.get("filterConfig", {}).get("filters", [])),
        ]

        for location_name, filter_list in filter_locations:
            if not isinstance(filter_list, list):
                continue

            for f in filter_list:
                filter_info = {
                    "source": source,
                    "name": f.get("name", f.get("displayName", "Unknown")),
                    "type": f.get("type", f.get("filterType", "Unknown")),
                }

                # Try to get target table/column
                if "expression" in f:
                    expr = f["expression"]
                    if isinstance(expr, dict):
                        col = expr.get("Column", {})
                        filter_info["table"] = col.get("Expression", {}).get("SourceRef", {}).get("Entity", "")
                        filter_info["column"] = col.get("Property", "")

                # Get filter values if available
                if "filter" in f:
                    filter_def = f["filter"]
                    if isinstance(filter_def, dict):
                        filter_info["condition"] = filter_def.get("Where", [{}])[0].get("Condition", {}) if "Where" in filter_def else {}

                filters.append(filter_info)

    except json.JSONDecodeError:
        pass

    return filters


@mcp.tool()
async def get_report_details(
    report: str,
    aspect: str = "all",
    workspace: Optional[str] = None,
    page_name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get details about a Power BI report's visuals, filters, and bookmarks.

    Combines visual inventory, filter configuration, and bookmark listings
    into a single tool with an aspect selector.

    Args:
        report: Report name or ID
        aspect: What to retrieve:
            'visuals' - Visual types, positions, and titles per page
            'filters' - Report/page/visual-level filters
            'bookmarks' - Saved bookmarks with target pages
            'all' - Everything
        workspace: Workspace name or ID (uses context if not provided)
        page_name: Filter by specific page (optional, for visuals/filters)
        ctx: Context object

    Returns:
        Report details based on requested aspect
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set."
        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        parsed = await _get_report_definition_parsed(
            fabric_client, powerbi_client, workspace_id, report
        )

        report_name = parsed["report_name"]
        page_parts = parsed["page_parts"]
        page_names = parsed["page_names"]
        config_parts = parsed["config_parts"]
        other_parts = parsed["other_parts"]

        markdown = ""

        # --- Visuals ---
        if aspect in ("visuals", "all"):
            filtered_page_parts = page_parts
            if page_name:
                filtered_page_parts = [(p, c) for p, c in page_parts if f"/pages/{page_name}/" in p]
                if not filtered_page_parts:
                    available = ", ".join(sorted(page_names)[:10])
                    if aspect == "visuals":
                        return f"Error: Page '{page_name}' not found. Available pages: {available}"

            markdown += f"# Visuals in Report: {report_name}\n\n"
            pages_visuals = {}
            for path, content in filtered_page_parts:
                pn = path.split("/pages/")[-1].split("/")[0] if "/pages/" in path else "Unknown"
                if not (path.endswith("page.json") or path.endswith(".json")):
                    continue
                visuals = _parse_visual_from_json(content)
                if visuals:
                    if pn not in pages_visuals:
                        pages_visuals[pn] = []
                    pages_visuals[pn].extend(visuals)

            if pages_visuals:
                total_visuals = 0
                for pn in sorted(pages_visuals.keys()):
                    visuals = pages_visuals[pn]
                    total_visuals += len(visuals)
                    markdown += f"## Page: {pn}\n\n"
                    markdown += "| Name | Type | Title | Position (x,y) | Size (w×h) |\n"
                    markdown += "|------|------|-------|----------------|------------|\n"
                    for v in visuals:
                        name = v.get("name", "N/A")
                        vtype = v.get("type", "Unknown")
                        title = v.get("title", "-")
                        x, y = v.get("x", 0), v.get("y", 0)
                        w, h = v.get("width", 0), v.get("height", 0)
                        markdown += f"| {name} | {vtype} | {title} | ({x}, {y}) | {w}×{h} |\n"
                    markdown += f"\n*{len(visuals)} visual(s) on this page*\n\n"
                    if len(markdown) > MAX_RESPONSE_SIZE - 500:
                        markdown += "\n*[Response size limit reached]*\n"
                        break
                markdown += f"\n---\n**Total:** {total_visuals} visual(s) across {len(pages_visuals)} page(s)\n\n"
            else:
                markdown += "*No visuals found.*\n\n"

        # --- Filters ---
        if aspect in ("filters", "all"):
            markdown += f"# Filters in Report: {report_name}\n\n"
            all_filters = {"report": [], "page": {}, "visual": {}}

            for path, content in config_parts:
                filters = _parse_filters_from_json(content, "report")
                all_filters["report"].extend(filters)

            for path, content in page_parts:
                pn = path.split("/pages/")[-1].split("/")[0] if "/pages/" in path else "Unknown"
                if page_name and pn != page_name:
                    continue
                filters = _parse_filters_from_json(content, f"page:{pn}")
                for f in filters:
                    if "visual" in f.get("source", "").lower():
                        if pn not in all_filters["visual"]:
                            all_filters["visual"][pn] = []
                        all_filters["visual"][pn].append(f)
                    else:
                        if pn not in all_filters["page"]:
                            all_filters["page"][pn] = []
                        all_filters["page"][pn].append(f)

            if all_filters["report"]:
                markdown += "## Report-Level Filters\n\n"
                markdown += "| Name | Type | Table | Column |\n|------|------|-------|--------|\n"
                for f in all_filters["report"]:
                    markdown += f"| {f.get('name', 'N/A')} | {f.get('type', 'N/A')} | {f.get('table', '-')} | {f.get('column', '-')} |\n"
                markdown += f"\n*{len(all_filters['report'])} report-level filter(s)*\n\n"

            if all_filters["page"]:
                markdown += "## Page-Level Filters\n\n"
                for pn, filters in sorted(all_filters["page"].items()):
                    markdown += f"### Page: {pn}\n\n"
                    markdown += "| Name | Type | Table | Column |\n|------|------|-------|--------|\n"
                    for f in filters:
                        markdown += f"| {f.get('name', 'N/A')} | {f.get('type', 'N/A')} | {f.get('table', '-')} | {f.get('column', '-')} |\n"
                    markdown += f"\n*{len(filters)} filter(s) on this page*\n\n"

            if all_filters["visual"]:
                markdown += "## Visual-Level Filters\n\n"
                for pn, filters in sorted(all_filters["visual"].items()):
                    markdown += f"### Page: {pn}\n\n"
                    markdown += "| Visual | Name | Type | Table | Column |\n|--------|------|------|-------|--------|\n"
                    for f in filters:
                        visual = f.get("visual", "-")
                        markdown += f"| {visual} | {f.get('name', 'N/A')} | {f.get('type', 'N/A')} | {f.get('table', '-')} | {f.get('column', '-')} |\n"
                    markdown += f"\n*{len(filters)} visual filter(s) on this page*\n\n"

            total_filters = (
                len(all_filters["report"]) +
                sum(len(f) for f in all_filters["page"].values()) +
                sum(len(f) for f in all_filters["visual"].values())
            )
            if total_filters == 0:
                markdown += "*No filters found in the report definition.*\n\n"
            else:
                markdown += f"\n---\n**Total:** {total_filters} filter(s) found\n\n"

        # --- Bookmarks ---
        if aspect in ("bookmarks", "all"):
            markdown += f"# Bookmarks in Report: {report_name}\n\n"
            bookmarks = []
            all_parts = config_parts + other_parts

            for path, content in all_parts:
                if content == "[Binary content]":
                    continue
                try:
                    data = json.loads(content)
                    bookmark_list = data.get("bookmarks", [])
                    if not bookmark_list and "config" in data:
                        config = data.get("config", {})
                        if isinstance(config, str):
                            try:
                                config = json.loads(config)
                            except Exception:
                                config = {}
                        bookmark_list = config.get("bookmarks", [])

                    for bm in bookmark_list:
                        bookmark = {
                            "name": bm.get("name", bm.get("displayName", "Unknown")),
                            "displayName": bm.get("displayName", bm.get("name", "")),
                            "targetPage": bm.get("explorationState", {}).get("activeSection", "-"),
                            "type": "Personal" if bm.get("personal", False) else "Report",
                        }
                        bookmarks.append(bookmark)
                except json.JSONDecodeError:
                    continue

            if not bookmarks:
                markdown += "*No bookmarks found in the report definition.*\n\n"
            else:
                markdown += "| Name | Display Name | Target Page | Type |\n"
                markdown += "|------|--------------|-------------|------|\n"
                for bm in bookmarks:
                    markdown += f"| {bm['name']} | {bm['displayName']} | {bm['targetPage']} | {bm['type']} |\n"
                markdown += f"\n*{len(bookmarks)} bookmark(s) found*\n\n"

        if not markdown:
            return f"Error: Unknown aspect '{aspect}'. Use 'visuals', 'filters', 'bookmarks', or 'all'."

        return markdown

    except Exception as e:
        logger.error(f"Error getting report details: {e}")
        return f"Error getting report details: {str(e)}"


@mcp.tool()
async def get_report_definition(
    report: str,
    workspace: Optional[str] = None,
    page_name: Optional[str] = None,
    summary_only: bool = False,
    ctx: Context = None
) -> str:
    """Get the structure/definition of a Power BI report.

    Returns information about the report's internal structure including
    pages, visuals, filters, and configuration. Uses the Fabric API
    getDefinition endpoint.

    Args:
        report: Report name or ID
        workspace: Workspace name or ID (uses context if not provided)
        page_name: Specific page name to retrieve (optional, retrieves all if not specified)
        summary_only: If True, returns only a summary of the report structure without full content
        ctx: Context object

    Returns:
        Report definition including:
        - Pages and their layout
        - Visuals on each page
        - Filters (report, page, visual level)
        - Configuration settings

    Note:
        - Requires read access to the report definition
        - Returns PBIR (Power BI Report) definition format
        - For large reports, use summary_only=True first to see available pages
        - Then use page_name parameter to retrieve specific page details

    Examples:
        get_report_definition("Sales Dashboard", summary_only=True)
        get_report_definition("Sales Dashboard", page_name="ReportSection1")
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

        # Extract page names for summary
        page_names = []
        for path, _ in page_parts:
            if "/pages/" in path:
                extracted_name = path.split("/pages/")[-1].split("/")[0]
                if extracted_name not in page_names:
                    page_names.append(extracted_name)

        # If summary_only, return just the structure overview
        if summary_only:
            markdown += "## Summary\n\n"
            markdown += f"- **Configuration files:** {len(config_parts)}\n"
            markdown += f"- **Page files:** {len(page_parts)}\n"
            markdown += f"- **Other files:** {len(other_parts)}\n"

            if page_names:
                markdown += f"\n### Pages ({len(page_names)} total)\n\n"
                for pn in sorted(page_names):
                    markdown += f"- `{pn}`\n"

            if config_parts:
                markdown += f"\n### Configuration Files\n\n"
                for path, _ in config_parts:
                    markdown += f"- {path}\n"

            if other_parts:
                markdown += f"\n### Other Files\n\n"
                for path, _ in other_parts[:20]:
                    markdown += f"- {path}\n"
                if len(other_parts) > 20:
                    markdown += f"- ... and {len(other_parts) - 20} more\n"

            markdown += f"\n---\n*Use `page_name` parameter to get details for a specific page.*"
            return markdown

        # Filter pages if page_name is specified
        if page_name:
            filtered_page_parts = [(path, content) for path, content in page_parts
                                   if f"/pages/{page_name}/" in path or path.endswith(f"/pages/{page_name}")]
            if not filtered_page_parts:
                available_pages = ", ".join(sorted(page_names)[:10])
                more_msg = f" (and {len(page_names) - 10} more)" if len(page_names) > 10 else ""
                return f"Error: Page '{page_name}' not found. Available pages: {available_pages}{more_msg}"
            page_parts = filtered_page_parts
            markdown += f"*Showing page: {page_name}*\n\n"

        # Track response size to avoid exceeding limits
        def truncate_to_fit(text: str, current_size: int, max_content: int = 5000) -> str:
            """Truncate content to fit within response limits."""
            remaining = MAX_RESPONSE_SIZE - current_size - 500  # Leave buffer
            max_allowed = min(remaining, max_content)
            if len(text) > max_allowed:
                return text[:max_allowed] + f"\n... [truncated, {len(text) - max_allowed} chars omitted]"
            return text

        # Format configuration (only include if not filtering by page)
        if config_parts and not page_name:
            markdown += "## Report Configuration\n\n"
            for path, content in config_parts:
                if len(markdown) > MAX_RESPONSE_SIZE - 1000:
                    markdown += "\n*[Response size limit reached. Use page_name or summary_only parameters.]*\n"
                    break
                markdown += f"### {path}\n\n"
                truncated_content = truncate_to_fit(content, len(markdown), 3000)
                markdown += f"```json\n{truncated_content}\n```\n\n"

        # Format pages
        if page_parts:
            if not page_name:
                markdown += "## Pages\n\n"
            for path, content in sorted(page_parts):
                if len(markdown) > MAX_RESPONSE_SIZE - 1000:
                    remaining_pages = len([p for p in page_parts if p[0] > path])
                    markdown += f"\n*[Response size limit reached. {remaining_pages} page(s) omitted. Use page_name parameter to view specific pages.]*\n"
                    break
                pn = path.split("/pages/")[-1].split("/")[0] if "/pages/" in path else path
                file_name = path.split("/")[-1] if "/" in path else path
                markdown += f"### Page: {pn} ({file_name})\n\n"
                truncated_content = truncate_to_fit(content, len(markdown), 4000)
                markdown += f"```json\n{truncated_content}\n```\n\n"

        # Summary (only if not filtered)
        if not page_name:
            markdown += f"\n## Summary\n\n"
            markdown += f"- **Configuration files:** {len(config_parts)}\n"
            markdown += f"- **Page files:** {len(page_parts)}\n"
            markdown += f"- **Other files:** {len(other_parts)}\n"

            if other_parts and len(markdown) < MAX_RESPONSE_SIZE - 500:
                markdown += f"\n### Other Files\n\n"
                for path, _ in other_parts[:10]:
                    markdown += f"- {path}\n"
                if len(other_parts) > 10:
                    markdown += f"- ... and {len(other_parts) - 10} more\n"

        # Final size check
        if len(markdown) > MAX_RESPONSE_SIZE:
            markdown = markdown[:MAX_RESPONSE_SIZE - 100] + f"\n\n*[Response truncated at {MAX_RESPONSE_SIZE} chars]*"

        return markdown

    except Exception as e:
        logger.error(f"Error getting report definition: {e}")
        return f"Error getting report definition: {str(e)}"
