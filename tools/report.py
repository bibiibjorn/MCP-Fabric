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

# Maximum response size to avoid "Tool result is too large" errors
MAX_RESPONSE_SIZE = 80000


def _parse_visual_from_json(content: str) -> List[Dict]:
    """Extract visual information from page JSON content."""
    visuals = []
    try:
        data = json.loads(content)
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
                config = vc.get("config", {})
                if isinstance(config, str):
                    try:
                        config = json.loads(config)
                    except:
                        pass
                if isinstance(config, dict):
                    visual["type"] = config.get("singleVisual", {}).get("visualType", "Unknown")
                    title_config = config.get("singleVisual", {}).get("vcObjects", {}).get("title", [])
                    if title_config and isinstance(title_config, list) and len(title_config) > 0:
                        title_props = title_config[0].get("properties", {})
                        title_text = title_props.get("text", {}).get("expr", {}).get("Literal", {}).get("Value", "")
                        if title_text:
                            visual["title"] = title_text.strip("'\"")
                visuals.append(visual)
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
                if "expression" in f:
                    expr = f["expression"]
                    if isinstance(expr, dict):
                        col = expr.get("Column", {})
                        filter_info["table"] = col.get("Expression", {}).get("SourceRef", {}).get("Entity", "")
                        filter_info["column"] = col.get("Property", "")
                if "filter" in f:
                    filter_def = f["filter"]
                    if isinstance(filter_def, dict):
                        filter_info["condition"] = filter_def.get("Where", [{}])[0].get("Condition", {}) if "Where" in filter_def else {}
                filters.append(filter_info)
    except json.JSONDecodeError:
        pass
    return filters


async def _get_report_definition_parsed(
    fabric_client: FabricApiClient,
    powerbi_client: PowerBIClient,
    workspace_id: str,
    report: str
) -> Dict:
    """Helper to fetch and parse report definition."""
    report_id = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)
    reports = await fabric_client.get_reports(workspace_id)
    report_info = next((r for r in reports if r.get("id") == report_id), {})
    report_name = report_info.get("displayName", report)

    definition = await fabric_client._make_request(
        f"workspaces/{workspace_id}/reports/{report_id}/getDefinition",
        method="POST", lro=True, lro_poll_interval=2, lro_timeout=120
    )
    if not definition:
        raise ValueError(f"Could not retrieve definition for report '{report_name}'")

    parts = definition.get("definition", {}).get("parts", [])
    config_parts, page_parts, other_parts = [], [], []

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

    page_names = []
    for path, _ in page_parts:
        if "/pages/" in path:
            extracted = path.split("/pages/")[-1].split("/")[0]
            if extracted not in page_names:
                page_names.append(extracted)

    return {
        "report_id": report_id, "report_name": report_name,
        "config_parts": config_parts, "page_parts": page_parts,
        "other_parts": other_parts, "page_names": page_names
    }


@mcp.tool()
async def manage_report(
    action: str,
    workspace: Optional[str] = None,
    report: Optional[str] = None,
    report_id: Optional[str] = None,
    aspect: Optional[str] = "all",
    page_name: Optional[str] = None,
    summary_only: bool = False,
    ctx: Context = None,
) -> str:
    """Manage Power BI reports - list, get details, pages, visuals, filters, and definitions.

    Args:
        action: Operation to perform:
            'list' - List all reports in a workspace
            'get' - Get a specific report by ID
            'get_pages' - List all pages in a report
            'get_details' - Get visuals, filters, and/or bookmarks
            'get_definition' - Get full report structure/definition
        workspace: Workspace name or ID (uses context if not provided)
        report: Report name or ID (required for get_pages, get_details, get_definition)
        report_id: Report ID (for 'get' action)
        aspect: For 'get_details': 'visuals', 'filters', 'bookmarks', or 'all'
        page_name: Filter by specific page (for get_details, get_definition)
        summary_only: For 'get_definition': return only structure overview
        ctx: Context object

    Returns:
        Report listings, details, pages, visuals, filters, or definitions
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set."

        if action == "list":
            client = ReportClient(fabric_client)
            reports = await client.list_reports(ws)
            if not reports:
                return f"No reports found in workspace '{ws}'."
            markdown = f"# Reports in workspace '{ws}'\n\n"
            markdown += "| ID | Name | Description |\n|-----|------|-------------|\n"
            for r in reports:
                markdown += f"| {r.get('id', 'N/A')} | {r.get('displayName', 'N/A')} | {r.get('description', 'N/A')} |\n"
            markdown += f"\n*{len(reports)} report(s) found*"
            return markdown

        elif action == "get":
            rid = report_id or report
            if not rid:
                return "Error: report or report_id is required for 'get' action."
            client = ReportClient(fabric_client)
            result = await client.get_report(ws, rid)
            if not result:
                return f"No report found with ID '{rid}' in workspace '{ws}'."
            return f"Report details:\n\n{result}"

        elif action == "get_pages":
            if not report:
                return "Error: 'report' is required for 'get_pages' action."
            powerbi_client = PowerBIClient(credential)
            workspace_id = await fabric_client.resolve_workspace(ws)
            rid = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)
            reports_list = await fabric_client.get_reports(workspace_id)
            report_info = next((r for r in reports_list if r.get("id") == rid), {})
            report_name = report_info.get("displayName", report)
            pages = await powerbi_client.get_report_pages(workspace_id, rid)
            if not pages:
                return f"No pages found in report '{report_name}'"
            markdown = f"# Pages in Report: {report_name}\n\n"
            markdown += "| Order | Name | Display Name |\n|-------|------|-------------|\n"
            for page in sorted(pages, key=lambda p: p.get("order", 0)):
                order = page.get("order", "N/A")
                name = page.get("name", "N/A")
                display_name = page.get("displayName", name)
                markdown += f"| {order} | {name} | {display_name} |\n"
            markdown += f"\n*{len(pages)} page(s) found*"
            return markdown

        elif action == "get_details":
            if not report:
                return "Error: 'report' is required for 'get_details' action."
            powerbi_client = PowerBIClient(credential)
            workspace_id = await fabric_client.resolve_workspace(ws)
            parsed = await _get_report_definition_parsed(fabric_client, powerbi_client, workspace_id, report)
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
                        markdown += "| Name | Type | Title | Position (x,y) | Size (w*h) |\n"
                        markdown += "|------|------|-------|----------------|------------|\n"
                        for v in visuals:
                            name = v.get("name", "N/A")
                            vtype = v.get("type", "Unknown")
                            title = v.get("title", "-")
                            x, y = v.get("x", 0), v.get("y", 0)
                            w, h = v.get("width", 0), v.get("height", 0)
                            markdown += f"| {name} | {vtype} | {title} | ({x}, {y}) | {w}x{h} |\n"
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
                    markdown += "## Report-Level Filters\n\n| Name | Type | Table | Column |\n|------|------|-------|--------|\n"
                    for f in all_filters["report"]:
                        markdown += f"| {f.get('name', 'N/A')} | {f.get('type', 'N/A')} | {f.get('table', '-')} | {f.get('column', '-')} |\n"
                if all_filters["page"]:
                    markdown += "\n## Page-Level Filters\n\n"
                    for pn, filters in sorted(all_filters["page"].items()):
                        markdown += f"### Page: {pn}\n\n| Name | Type | Table | Column |\n|------|------|-------|--------|\n"
                        for f in filters:
                            markdown += f"| {f.get('name', 'N/A')} | {f.get('type', 'N/A')} | {f.get('table', '-')} | {f.get('column', '-')} |\n"
                total_filters = len(all_filters["report"]) + sum(len(f) for f in all_filters["page"].values()) + sum(len(f) for f in all_filters["visual"].values())
                if total_filters == 0:
                    markdown += "*No filters found.*\n\n"

            # --- Bookmarks ---
            if aspect in ("bookmarks", "all"):
                markdown += f"# Bookmarks in Report: {report_name}\n\n"
                bookmarks = []
                for path, content in config_parts + other_parts:
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
                            bookmarks.append({
                                "name": bm.get("name", bm.get("displayName", "Unknown")),
                                "displayName": bm.get("displayName", bm.get("name", "")),
                                "targetPage": bm.get("explorationState", {}).get("activeSection", "-"),
                                "type": "Personal" if bm.get("personal", False) else "Report",
                            })
                    except json.JSONDecodeError:
                        continue
                if not bookmarks:
                    markdown += "*No bookmarks found.*\n\n"
                else:
                    markdown += "| Name | Display Name | Target Page | Type |\n|------|--------------|-------------|------|\n"
                    for bm in bookmarks:
                        markdown += f"| {bm['name']} | {bm['displayName']} | {bm['targetPage']} | {bm['type']} |\n"
                    markdown += f"\n*{len(bookmarks)} bookmark(s) found*\n\n"

            if not markdown:
                return f"Error: Unknown aspect '{aspect}'. Use 'visuals', 'filters', 'bookmarks', or 'all'."
            return markdown

        elif action == "get_definition":
            if not report:
                return "Error: 'report' is required for 'get_definition' action."
            powerbi_client = PowerBIClient(credential)
            workspace_id = await fabric_client.resolve_workspace(ws)
            rid = await powerbi_client.resolve_report_id(workspace_id, report, fabric_client)
            reports_list = await fabric_client.get_reports(workspace_id)
            report_info = next((r for r in reports_list if r.get("id") == rid), {})
            report_name = report_info.get("displayName", report)

            definition = await fabric_client._make_request(
                f"workspaces/{workspace_id}/reports/{rid}/getDefinition",
                method="POST", lro=True, lro_poll_interval=2, lro_timeout=120
            )
            if not definition:
                return f"Error: Could not retrieve definition for report '{report_name}'"

            parts = definition.get("definition", {}).get("parts", [])
            if not parts:
                return f"No definition parts found for '{report_name}'"

            config_parts, page_parts_def, other_parts_def = [], [], []
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
                    page_parts_def.append((path, content))
                else:
                    other_parts_def.append((path, content))

            page_names_def = []
            for path, _ in page_parts_def:
                if "/pages/" in path:
                    extracted_name = path.split("/pages/")[-1].split("/")[0]
                    if extracted_name not in page_names_def:
                        page_names_def.append(extracted_name)

            markdown = f"# Report Definition: {report_name}\n\n"

            if summary_only:
                markdown += "## Summary\n\n"
                markdown += f"- **Configuration files:** {len(config_parts)}\n"
                markdown += f"- **Page files:** {len(page_parts_def)}\n"
                markdown += f"- **Other files:** {len(other_parts_def)}\n"
                if page_names_def:
                    markdown += f"\n### Pages ({len(page_names_def)} total)\n\n"
                    for pn in sorted(page_names_def):
                        markdown += f"- `{pn}`\n"
                markdown += f"\n*Use page_name parameter to get details for a specific page.*"
                return markdown

            if page_name:
                filtered = [(p, c) for p, c in page_parts_def if f"/pages/{page_name}/" in p or p.endswith(f"/pages/{page_name}")]
                if not filtered:
                    available = ", ".join(sorted(page_names_def)[:10])
                    return f"Error: Page '{page_name}' not found. Available pages: {available}"
                page_parts_def = filtered
                markdown += f"*Showing page: {page_name}*\n\n"

            def truncate_to_fit(text, current_size, max_content=5000):
                remaining = MAX_RESPONSE_SIZE - current_size - 500
                max_allowed = min(remaining, max_content)
                if len(text) > max_allowed:
                    return text[:max_allowed] + f"\n... [truncated, {len(text) - max_allowed} chars omitted]"
                return text

            if config_parts and not page_name:
                markdown += "## Report Configuration\n\n"
                for path, content in config_parts:
                    if len(markdown) > MAX_RESPONSE_SIZE - 1000:
                        markdown += "\n*[Response size limit reached]*\n"
                        break
                    markdown += f"### {path}\n\n```json\n{truncate_to_fit(content, len(markdown), 3000)}\n```\n\n"

            if page_parts_def:
                if not page_name:
                    markdown += "## Pages\n\n"
                for path, content in sorted(page_parts_def):
                    if len(markdown) > MAX_RESPONSE_SIZE - 1000:
                        markdown += f"\n*[Response size limit reached. Use page_name parameter.]*\n"
                        break
                    pn = path.split("/pages/")[-1].split("/")[0] if "/pages/" in path else path
                    file_name = path.split("/")[-1] if "/" in path else path
                    markdown += f"### Page: {pn} ({file_name})\n\n```json\n{truncate_to_fit(content, len(markdown), 4000)}\n```\n\n"

            if len(markdown) > MAX_RESPONSE_SIZE:
                markdown = markdown[:MAX_RESPONSE_SIZE - 100] + f"\n\n*[Response truncated]*"
            return markdown

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'get', 'get_pages', 'get_details', or 'get_definition'."

    except Exception as e:
        logger.error(f"Error managing report: {e}")
        return f"Error managing report: {str(e)}"
