from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    SemanticModelClient,
    PowerBIClient,
)
from helpers.logging_config import get_logger

from typing import Optional, List, Dict, Tuple, Any
import base64
import re

logger = get_logger(__name__)


# =============================================================================
# TMDL Parsing Utilities
# =============================================================================

def _parse_tmdl_measures(content: str) -> List[Dict[str, str]]:
    """Parse measures from TMDL table content."""
    measures = []
    lines = content.split('\n')
    current_measure = None
    expression_lines = []
    in_expression = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        measure_match = re.match(r"^\s*measure\s+'([^']+)'\s*=\s*(.*)", line)
        if measure_match:
            if current_measure:
                current_measure['expression'] = '\n'.join(expression_lines).strip()
                measures.append(current_measure)

            measure_name = measure_match.group(1)
            first_expr = measure_match.group(2).strip()
            current_measure = {
                'name': measure_name, 'expression': '',
                'description': '', 'formatString': '', 'displayFolder': ''
            }
            expression_lines = [first_expr] if first_expr else []
            in_expression = True
            i += 1
            continue

        if current_measure and in_expression:
            prop_match = re.match(r"^\s+(formatString|description|displayFolder)\s*[:=]\s*(.*)", line)
            if prop_match:
                prop_name = prop_match.group(1)
                prop_value = prop_match.group(2).strip()
                if prop_value.startswith('"') and prop_value.endswith('"'):
                    prop_value = prop_value[1:-1]
                elif prop_value.startswith("'") and prop_value.endswith("'"):
                    prop_value = prop_value[1:-1]
                elif prop_value.startswith('```'):
                    prop_value = prop_value[3:]
                    i += 1
                    while i < len(lines) and not lines[i].strip().endswith('```'):
                        prop_value += '\n' + lines[i]
                        i += 1
                    if i < len(lines):
                        prop_value += '\n' + lines[i].strip()[:-3]
                current_measure[prop_name] = prop_value
                i += 1
                continue

            if stripped and not line.startswith('\t') and not line.startswith('    '):
                current_measure['expression'] = '\n'.join(expression_lines).strip()
                measures.append(current_measure)
                current_measure = None
                in_expression = False
                continue

            if stripped and (line.startswith('\t') or line.startswith('    ')):
                expr_line = line
                if line.startswith('\t\t'):
                    expr_line = line[2:]
                elif line.startswith('        '):
                    expr_line = line[8:]
                elif line.startswith('\t'):
                    expr_line = line[1:]
                elif line.startswith('    '):
                    expr_line = line[4:]
                expression_lines.append(expr_line.rstrip())

        i += 1

    if current_measure:
        current_measure['expression'] = '\n'.join(expression_lines).strip()
        measures.append(current_measure)

    return measures


def _parse_tmdl_columns(content: str) -> List[Dict[str, str]]:
    """Parse columns from TMDL table content."""
    columns = []
    lines = content.split('\n')
    current_column = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        col_match = re.match(r"^\s*column\s+'([^']+)'", line)
        if col_match:
            if current_column:
                columns.append(current_column)
            current_column = {
                'name': col_match.group(1), 'dataType': '',
                'sourceColumn': '', 'isHidden': False, 'description': ''
            }
            i += 1
            continue

        if current_column:
            if stripped.startswith('dataType:'):
                current_column['dataType'] = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('sourceColumn:'):
                val = stripped.split(':', 1)[1].strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                current_column['sourceColumn'] = val
            elif stripped == 'isHidden':
                current_column['isHidden'] = True
            elif stripped.startswith('description:'):
                val = stripped.split(':', 1)[1].strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                current_column['description'] = val

            if stripped and not line.startswith('\t') and not line.startswith('    '):
                if not col_match:
                    columns.append(current_column)
                    current_column = None

        i += 1

    if current_column:
        columns.append(current_column)

    return columns


def _parse_tmdl_table_name(content: str) -> str:
    """Extract table name from TMDL content."""
    match = re.search(r"^table\s+'([^']+)'", content, re.MULTILINE)
    return match.group(1) if match else "Unknown"


async def _get_model_definition_parts(
    fabric_client, powerbi_client, workspace_id: str, dataset_id: str
) -> List[Tuple[str, str]]:
    """Get and decode TMDL parts from semantic model definition."""
    definition = await fabric_client._make_request(
        f"workspaces/{workspace_id}/semanticModels/{dataset_id}/getDefinition",
        method="POST", lro=True, lro_poll_interval=2, lro_timeout=120
    )
    if not definition:
        return []
    parts = definition.get("definition", {}).get("parts", [])
    result = []
    for part in parts:
        path = part.get("path", "")
        payload = part.get("payload", "")
        if not payload:
            continue
        try:
            content = base64.b64decode(payload).decode("utf-8")
            result.append((path, content))
        except Exception:
            continue
    return result


def _format_dax_results_as_markdown(results: Dict) -> str:
    """Format DAX query results as a markdown table."""
    if not results or "results" not in results:
        return "No results returned"
    tables = results.get("results", [])
    if not tables:
        return "Query returned no data"

    markdown_parts = []
    for idx, table in enumerate(tables):
        rows = table.get("tables", [{}])[0].get("rows", [])
        if not rows:
            markdown_parts.append(f"*Table {idx + 1}: No rows*\n")
            continue
        columns = list(rows[0].keys())
        header = "| " + " | ".join(str(col) for col in columns) + " |"
        separator = "|" + "|".join("---" for _ in columns) + "|"
        data_rows = []
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    val = f"{val:,.2f}" if val != int(val) else f"{int(val):,}"
                elif isinstance(val, int):
                    val = f"{val:,}"
                elif val is None:
                    val = ""
                values.append(str(val))
            data_rows.append("| " + " | ".join(values) + " |")
        table_md = "\n".join([header, separator] + data_rows)
        if len(tables) > 1:
            markdown_parts.append(f"### Table {idx + 1}\n\n{table_md}\n")
        else:
            markdown_parts.append(table_md)

    row_count = sum(len(t.get("tables", [{}])[0].get("rows", [])) for t in tables)
    markdown_parts.append(f"\n*{row_count} row(s) returned*")
    return "\n".join(markdown_parts)


def _escape_dax_string(value: str) -> str:
    """Escape a string value for use in DAX."""
    return value.replace('"', '""')


def _normalize_column_reference(column_ref: str) -> str:
    """Normalize a column reference to proper DAX format."""
    column_ref = column_ref.strip()
    if re.match(r"^'[^']+'\[[^\]]+\]$", column_ref):
        return column_ref
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\[([^\]]+)\]$", column_ref)
    if match:
        table, col = match.groups()
        return f"'{table}'[{col}]"
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\.([A-Za-z_][A-Za-z0-9_ ]*)$", column_ref)
    if match:
        table, col = match.groups()
        return f"'{table}'[{col}]"
    if re.match(r"^\[[^\]]+\]$", column_ref):
        return column_ref
    if re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*$", column_ref):
        return f"[{column_ref}]"
    return column_ref


def _build_summarize_columns_query(
    measures: List[str], group_by: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[str]]] = None, max_rows: int = 1000
) -> str:
    """Build a SUMMARIZECOLUMNS DAX query."""
    if group_by:
        normalized_groups = [_normalize_column_reference(col) for col in group_by]
        group_columns = ", ".join(normalized_groups)
    else:
        group_columns = ""

    measure_exprs = []
    for measure in measures:
        if "[" in measure and "]" in measure:
            measure_exprs.append(f'"{measure}", {measure}')
        else:
            measure_exprs.append(f'"{measure}", [{measure}]')
    measures_str = ", ".join(measure_exprs)

    filter_str = ""
    if filters:
        filter_exprs = []
        for column, values in filters.items():
            normalized_col = _normalize_column_reference(column)
            if len(values) == 1:
                filter_exprs.append(f'{normalized_col} = "{_escape_dax_string(values[0])}"')
            else:
                val_list = ", ".join(f'"{_escape_dax_string(v)}"' for v in values)
                filter_exprs.append(f'{normalized_col} IN {{{val_list}}}')
        filter_str = ", " + ", ".join(filter_exprs)

    if group_columns:
        return f"EVALUATE TOPN({max_rows}, SUMMARIZECOLUMNS({group_columns}, {measures_str}{filter_str}))"
    else:
        return f"EVALUATE ROW({measures_str})"


# =============================================================================
# Consolidated Tools
# =============================================================================

@mcp.tool()
async def manage_semantic_model(
    action: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    model_id: Optional[str] = None,
    aspect: Optional[str] = "all",
    search: Optional[str] = None,
    include_expression: bool = True,
    include_hidden: bool = False,
    top: int = 10,
    ctx: Context = None,
) -> str:
    """Manage Power BI semantic models - list, get details, metadata, and refresh history.

    Args:
        action: Operation to perform:
            'list' - List all semantic models in a workspace
            'get' - Get a specific semantic model by ID
            'get_metadata' - Get model structure (schema, measures, tables)
            'get_refresh_history' - Get refresh history with status and timing
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        model_id: Model ID (for 'get' action)
        aspect: For 'get_metadata': 'schema', 'measures', 'tables', or 'all'
        search: Filter measures by search term (for get_metadata)
        include_expression: Include DAX expressions for measures (default: True)
        include_hidden: Include hidden columns in tables (default: False)
        top: Number of refresh entries to return (for get_refresh_history, default: 10)
        ctx: Context object

    Returns:
        Model listings, details, metadata, or refresh history
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)

        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set."

        if action == "list":
            client = SemanticModelClient(fabric_client)
            models = await client.list_semantic_models(ws)
            if not models:
                return f"No semantic models found in workspace '{ws}'."
            markdown = f"# Semantic Models in workspace '{ws}'\n\n"
            markdown += "| ID | Name | Folder ID | Description |\n|-----|------|-----------|-------------|\n"
            for model in models:
                markdown += f"| {model.get('id', 'N/A')} | {model.get('displayName', 'N/A')} | {model.get('folderId', 'N/A')} | {model.get('description', 'N/A')} |\n"
            markdown += f"\n*{len(models)} semantic model(s) found*"
            return markdown

        elif action == "get":
            mid = model_id or dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
            if not mid:
                return "Error: No model_id or dataset specified and no semantic model context set."
            client = SemanticModelClient(fabric_client)
            model = await client.get_semantic_model(ws, mid)
            if not model:
                return f"No semantic model found with ID '{mid}' in workspace '{ws}'."
            return f"Semantic Model '{model.get('displayName', 'Unknown')}' details:\n\n{model}"

        elif action == "get_metadata":
            powerbi_client = PowerBIClient(credential)
            workspace_id = await fabric_client.resolve_workspace(ws)
            dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
            dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
            if not dataset_id:
                return "Error: No dataset specified and no semantic model context set."
            dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

            if not dataset_name or dataset_name == dataset_id:
                models = await fabric_client.get_semantic_models(workspace_id)
                model_info = next((m for m in models if m.get("id") == dataset_id), {})
                dataset_name = model_info.get("displayName", dataset_id)

            if aspect == "schema":
                definition = await fabric_client._make_request(
                    f"workspaces/{workspace_id}/semanticModels/{dataset_id}/getDefinition",
                    method="POST", lro=True, lro_poll_interval=2, lro_timeout=120,
                )
                if not definition:
                    return f"Error: Could not retrieve definition for '{dataset_name}'"
                parts = definition.get("definition", {}).get("parts", [])
                if not parts:
                    return f"No schema definition found for '{dataset_name}'"
                markdown = f"# Schema: {dataset_name}\n\n"
                tables_content, relationships_content = [], []
                for part in parts:
                    path = part.get("path", "")
                    payload = part.get("payload", "")
                    if not payload:
                        continue
                    try:
                        content = base64.b64decode(payload).decode("utf-8")
                    except Exception:
                        continue
                    if "/tables/" in path and path.endswith(".tmdl"):
                        tables_content.append((path, content))
                    elif "relationships.tmdl" in path:
                        relationships_content.append(content)
                if tables_content:
                    markdown += "## Tables\n\n"
                    for path, content in sorted(tables_content):
                        table_name = path.split("/tables/")[-1].replace(".tmdl", "") if "/tables/" in path else "Unknown"
                        markdown += f"### {table_name}\n\n```\n{content}\n```\n\n"
                if relationships_content:
                    markdown += "## Relationships\n\n"
                    for content in relationships_content:
                        markdown += f"```\n{content}\n```\n\n"
                markdown += f"\n*{len(tables_content)} table(s) found*"
                return markdown

            parts = await _get_model_definition_parts(fabric_client, powerbi_client, workspace_id, dataset_id)
            if not parts:
                return f"Error: Could not retrieve definition for '{dataset_name}'"
            markdown = ""

            if aspect in ("measures", "all"):
                all_measures = []
                for path, content in parts:
                    if "/tables/" in path and path.endswith(".tmdl"):
                        table_name = _parse_tmdl_table_name(content)
                        parsed_measures = _parse_tmdl_measures(content)
                        for measure in parsed_measures:
                            measure['table'] = table_name
                            all_measures.append(measure)
                if search:
                    search_lower = search.lower()
                    all_measures = [m for m in all_measures if search_lower in m['name'].lower() or search_lower in m.get('description', '').lower() or search_lower in m.get('displayFolder', '').lower()]
                if all_measures:
                    all_measures.sort(key=lambda m: (m.get('table', ''), m['name']))
                    markdown += f"# Measures in {dataset_name}\n\n"
                    if search:
                        markdown += f"*Filtered by: '{search}'*\n\n"
                    markdown += f"**{len(all_measures)} measure(s) found**\n\n"
                    current_table = None
                    for measure in all_measures:
                        table = measure.get('table', 'Unknown')
                        if table != current_table:
                            markdown += f"\n## Table: {table}\n\n"
                            current_table = table
                        markdown += f"### {measure['name']}\n\n"
                        if measure.get('description'):
                            markdown += f"*{measure['description']}*\n\n"
                        if measure.get('displayFolder'):
                            markdown += f"**Folder:** {measure['displayFolder']}\n"
                        if measure.get('formatString'):
                            markdown += f"**Format:** `{measure['formatString']}`\n"
                        if include_expression and measure.get('expression'):
                            expr = measure['expression']
                            markdown += f"\n```dax\n{expr}\n```\n\n"
                        else:
                            markdown += "\n"
                elif aspect == "measures":
                    return f"No measures found{f' matching {search!r}' if search else ''} in '{dataset_name}'"

            if aspect in ("tables", "all"):
                all_tables = []
                for path, content in parts:
                    if "/tables/" in path and path.endswith(".tmdl"):
                        table_name = _parse_tmdl_table_name(content)
                        columns = _parse_tmdl_columns(content)
                        if not include_hidden:
                            columns = [c for c in columns if not c.get('isHidden', False)]
                        all_tables.append({'name': table_name, 'columns': columns})
                if all_tables:
                    all_tables.sort(key=lambda t: t['name'])
                    markdown += f"\n# Tables in {dataset_name}\n\n"
                    markdown += f"**{len(all_tables)} table(s) found**\n\n"
                    for table in all_tables:
                        table_name = table['name']
                        columns = table['columns']
                        markdown += f"## {table_name}\n\n"
                        if not columns:
                            markdown += "*No visible columns*\n\n"
                            continue
                        markdown += "| Column | Data Type | Description |\n|--------|-----------|-------------|\n"
                        for col in columns:
                            col_ref = f"`'{table_name}'[{col['name']}]`"
                            markdown += f"| {col_ref} | {col.get('dataType', 'unknown')} | {col.get('description', '')} |\n"
                        markdown += "\n"
                elif aspect == "tables":
                    return f"No tables found in semantic model '{dataset_name}'"

            if not markdown:
                return f"Error: Unknown aspect '{aspect}'. Use 'schema', 'measures', 'tables', or 'all'."
            return markdown

        elif action == "get_refresh_history":
            powerbi_client = PowerBIClient(credential)
            workspace_id = await fabric_client.resolve_workspace(ws)
            dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
            dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
            if not dataset_id:
                return "Error: No dataset specified and no semantic model context set."
            dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

            if not dataset_name or dataset_name == dataset_id:
                models = await fabric_client.get_semantic_models(workspace_id)
                model_info = next((m for m in models if m.get("id") == dataset_id), {})
                dataset_name = model_info.get("displayName", dataset_id)

            refreshes = await powerbi_client.get_refresh_history(workspace_id, dataset_id, top)
            if not refreshes:
                return f"No refresh history found for semantic model '{dataset_name}'"

            markdown = f"# Refresh History: {dataset_name}\n\n"
            markdown += f"**{len(refreshes)} refresh(es) shown**\n\n"
            markdown += "| Status | Type | Start Time | End Time | Duration |\n|--------|------|------------|----------|----------|\n"

            for refresh in refreshes:
                status = refresh.get("status", "Unknown")
                refresh_type = refresh.get("refreshType", "Unknown")
                start_time = refresh.get("startTime", "N/A")
                end_time = refresh.get("endTime", "N/A")
                duration = "N/A"
                if start_time != "N/A" and end_time != "N/A":
                    try:
                        from datetime import datetime
                        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                        total_seconds = int((end_dt - start_dt).total_seconds())
                        if total_seconds < 60:
                            duration = f"{total_seconds}s"
                        elif total_seconds < 3600:
                            duration = f"{total_seconds // 60}m {total_seconds % 60}s"
                        else:
                            duration = f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m"
                    except Exception:
                        pass
                if start_time != "N/A":
                    start_time = start_time.replace("T", " ").replace("Z", " UTC")[:22]
                if end_time != "N/A":
                    end_time = end_time.replace("T", " ").replace("Z", " UTC")[:22]
                markdown += f"| {status} | {refresh_type} | {start_time} | {end_time} | {duration} |\n"

            failed = [r for r in refreshes if r.get("status") == "Failed"]
            if failed:
                markdown += "\n## Failed Refresh Details\n\n"
                for refresh in failed:
                    st = refresh.get("startTime", "Unknown time")
                    if st != "Unknown time":
                        st = st.replace("T", " ").replace("Z", " UTC")[:22]
                    markdown += f"### Refresh at {st}\n\n"
                    error_json = refresh.get("serviceExceptionJson")
                    if error_json:
                        try:
                            import json
                            error_data = json.loads(error_json)
                            markdown += f"**Error Code:** {error_data.get('errorCode', 'Unknown')}\n\n"
                            markdown += f"**Description:** {error_data.get('errorDescription', 'No description')}\n\n"
                        except Exception:
                            markdown += f"**Error:** {error_json}\n\n"
                    else:
                        markdown += "*No error details available*\n\n"
            return markdown

        else:
            return f"Error: Unknown action '{action}'. Use 'list', 'get', 'get_metadata', or 'get_refresh_history'."

    except Exception as e:
        logger.error(f"Error managing semantic model: {e}")
        return f"Error managing semantic model: {str(e)}"


@mcp.tool()
async def query_semantic_model(
    action: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    dax_query: Optional[str] = None,
    table_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    measures: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[str]]] = None,
    max_rows: int = 1000,
    ctx: Context = None
) -> str:
    """Query Power BI semantic models - execute DAX, read tables, evaluate measures.

    Args:
        action: Operation to perform:
            'execute_dax' - Execute a raw DAX query
            'read_table' - Read data from a table
            'evaluate_measure' - Evaluate measures with grouping and filters
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        dax_query: DAX query string (for 'execute_dax')
        table_name: Table name (for 'read_table')
        columns: Columns to return (for 'read_table', all if not specified)
        measures: Measure names to compute (for 'evaluate_measure')
        group_by: Columns to group by (for 'evaluate_measure'). Formats: "'Date'[Year]", "Date[Year]", "Date.Year"
        filters: Filter conditions as dict of column->values (for 'evaluate_measure')
        max_rows: Maximum rows to return (default 1000, max 100000)
        ctx: Context object

    Returns:
        Query results as formatted markdown tables
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set."
        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."
        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        max_rows = min(max(1, max_rows), 100000)

        if action == "execute_dax":
            if not dax_query:
                return "Error: 'dax_query' is required for 'execute_dax' action."
            query = dax_query.strip()
            if "TOPN" not in query.upper() and "ROW(" not in query.upper():
                if query.upper().startswith("EVALUATE"):
                    expr = query[8:].strip()
                    query = f"EVALUATE TOPN({max_rows}, {expr})"
                else:
                    query = f"EVALUATE TOPN({max_rows}, {query})"
            results = await powerbi_client.execute_dax(workspace_id, dataset_id, query)
            markdown = _format_dax_results_as_markdown(results)
            return f"## DAX Query Results\n\n**Query:** `{dax_query[:200]}{'...' if len(dax_query) > 200 else ''}`\n\n{markdown}"

        elif action == "read_table":
            if not table_name:
                return "Error: 'table_name' is required for 'read_table' action."
            table_ref = f"'{table_name}'"
            if columns:
                col_exprs = []
                for col in columns:
                    if "[" in col:
                        col_exprs.append(f'"{col}", {col}')
                    else:
                        col_exprs.append(f'"{col}", {table_ref}[{col}]')
                cols_str = ", ".join(col_exprs)
                dax = f"EVALUATE TOPN({max_rows}, SELECTCOLUMNS({table_ref}, {cols_str}))"
            else:
                dax = f"EVALUATE TOPN({max_rows}, {table_ref})"
            results = await powerbi_client.execute_dax(workspace_id, dataset_id, dax)
            markdown = _format_dax_results_as_markdown(results)
            return f"## Table: {table_name}\n\n{markdown}"

        elif action == "evaluate_measure":
            if not measures:
                return "Error: 'measures' list is required for 'evaluate_measure' action."
            dax = _build_summarize_columns_query(measures, group_by, filters, max_rows)
            results = await powerbi_client.execute_dax(workspace_id, dataset_id, dax)
            markdown = _format_dax_results_as_markdown(results)
            desc = f"**Measures:** {', '.join(measures)}"
            if group_by:
                desc += f"\n**Grouped by:** {', '.join(group_by)}"
            if filters:
                filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items())
                desc += f"\n**Filters:** {filter_desc}"
            return f"## Measure Evaluation\n\n{desc}\n\n{markdown}"

        else:
            return f"Error: Unknown action '{action}'. Use 'execute_dax', 'read_table', or 'evaluate_measure'."

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error querying semantic model: {e}")
        return f"Error querying semantic model: {str(e)}"
