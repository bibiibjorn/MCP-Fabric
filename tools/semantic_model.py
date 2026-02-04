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
    """Parse measures from TMDL table content.

    Args:
        content: TMDL file content for a table

    Returns:
        List of dicts with 'name', 'expression', 'description', 'formatString', 'displayFolder'
    """
    measures = []

    # TMDL measure format:
    # measure 'Measure Name' = expression
    #     formatString: "format"
    #     description: "desc"
    #     displayFolder: "folder"
    #
    # Multi-line expressions use indentation

    lines = content.split('\n')
    current_measure = None
    expression_lines = []
    in_expression = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for measure definition start
        measure_match = re.match(r"^\s*measure\s+'([^']+)'\s*=\s*(.*)", line)
        if measure_match:
            # Save previous measure if exists
            if current_measure:
                current_measure['expression'] = '\n'.join(expression_lines).strip()
                measures.append(current_measure)

            measure_name = measure_match.group(1)
            first_expr = measure_match.group(2).strip()

            current_measure = {
                'name': measure_name,
                'expression': '',
                'description': '',
                'formatString': '',
                'displayFolder': ''
            }
            expression_lines = [first_expr] if first_expr else []
            in_expression = True
            i += 1
            continue

        # If we're in a measure, check for properties or continuation
        if current_measure and in_expression:
            # Check for properties (they start with specific keywords)
            prop_match = re.match(r"^\s+(formatString|description|displayFolder)\s*[:=]\s*(.*)", line)
            if prop_match:
                prop_name = prop_match.group(1)
                prop_value = prop_match.group(2).strip()
                # Remove surrounding quotes if present
                if prop_value.startswith('"') and prop_value.endswith('"'):
                    prop_value = prop_value[1:-1]
                elif prop_value.startswith("'") and prop_value.endswith("'"):
                    prop_value = prop_value[1:-1]
                # Handle ```...``` multi-line descriptions
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

            # Check if this is a new definition (table, column, another measure)
            if stripped and not line.startswith('\t') and not line.startswith('    '):
                # End of current measure
                current_measure['expression'] = '\n'.join(expression_lines).strip()
                measures.append(current_measure)
                current_measure = None
                in_expression = False
                continue

            # It's part of the expression (indented continuation)
            if stripped and (line.startswith('\t') or line.startswith('    ')):
                # Remove one level of indentation
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

    # Don't forget the last measure
    if current_measure:
        current_measure['expression'] = '\n'.join(expression_lines).strip()
        measures.append(current_measure)

    return measures


def _parse_tmdl_columns(content: str) -> List[Dict[str, str]]:
    """Parse columns from TMDL table content.

    Args:
        content: TMDL file content for a table

    Returns:
        List of dicts with 'name', 'dataType', 'sourceColumn', 'isHidden', 'description'
    """
    columns = []

    # TMDL column format:
    # column 'Column Name'
    #     dataType: string
    #     sourceColumn: "SourceColumnName"
    #     isHidden

    lines = content.split('\n')
    current_column = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for column definition start
        col_match = re.match(r"^\s*column\s+'([^']+)'", line)
        if col_match:
            # Save previous column if exists
            if current_column:
                columns.append(current_column)

            col_name = col_match.group(1)
            current_column = {
                'name': col_name,
                'dataType': '',
                'sourceColumn': '',
                'isHidden': False,
                'description': ''
            }
            i += 1
            continue

        # If we're in a column, check for properties
        if current_column:
            # Check for properties
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

            # Check if we hit a new definition
            if stripped and not line.startswith('\t') and not line.startswith('    '):
                if not col_match:  # Not a continuation of current
                    columns.append(current_column)
                    current_column = None

        i += 1

    # Don't forget the last column
    if current_column:
        columns.append(current_column)

    return columns


def _parse_tmdl_table_name(content: str) -> str:
    """Extract table name from TMDL content."""
    match = re.search(r"^table\s+'([^']+)'", content, re.MULTILINE)
    if match:
        return match.group(1)
    return "Unknown"


async def _get_model_definition_parts(
    fabric_client,
    powerbi_client,
    workspace_id: str,
    dataset_id: str
) -> List[Tuple[str, str]]:
    """Get and decode TMDL parts from semantic model definition.

    Returns:
        List of (path, content) tuples
    """
    definition = await fabric_client._make_request(
        f"workspaces/{workspace_id}/semanticModels/{dataset_id}/getDefinition",
        method="POST",
        lro=True,
        lro_poll_interval=2,
        lro_timeout=120
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
    """Format DAX query results as a markdown table.

    Args:
        results: The results dictionary from execute_dax

    Returns:
        Markdown formatted string
    """
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

        # Get columns from first row
        columns = list(rows[0].keys())

        # Build markdown table
        header = "| " + " | ".join(str(col) for col in columns) + " |"
        separator = "|" + "|".join("---" for _ in columns) + "|"

        data_rows = []
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, "")
                # Format numbers nicely
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

    row_count = sum(
        len(t.get("tables", [{}])[0].get("rows", []))
        for t in tables
    )
    markdown_parts.append(f"\n*{row_count} row(s) returned*")

    return "\n".join(markdown_parts)


def _escape_dax_string(value: str) -> str:
    """Escape a string value for use in DAX."""
    return value.replace('"', '""')


def _normalize_column_reference(column_ref: str) -> str:
    """Normalize a column reference to proper DAX format.

    Accepts multiple formats:
    - 'Table'[Column] - already proper DAX format, return as-is
    - Table[Column] - add quotes around table name
    - Table.Column - convert to 'Table'[Column]
    - [Column] - leave as-is (unqualified)

    Args:
        column_ref: Column reference in any supported format

    Returns:
        Properly formatted DAX column reference
    """
    column_ref = column_ref.strip()

    # Already proper DAX format with quoted table
    if re.match(r"^'[^']+'\[[^\]]+\]$", column_ref):
        return column_ref

    # Unquoted table with brackets: Table[Column] -> 'Table'[Column]
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\[([^\]]+)\]$", column_ref)
    if match:
        table, col = match.groups()
        return f"'{table}'[{col}]"

    # Dot notation: Table.Column -> 'Table'[Column]
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\.([A-Za-z_][A-Za-z0-9_ ]*)$", column_ref)
    if match:
        table, col = match.groups()
        return f"'{table}'[{col}]"

    # Unqualified column reference: [Column] - return as-is
    if re.match(r"^\[[^\]]+\]$", column_ref):
        return column_ref

    # Just a column name without any brackets - wrap in brackets
    if re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*$", column_ref):
        return f"[{column_ref}]"

    # Unknown format - return as-is and let DAX engine handle it
    return column_ref


def _build_summarize_columns_query(
    measures: List[str],
    group_by: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[str]]] = None,
    max_rows: int = 1000
) -> str:
    """Build a SUMMARIZECOLUMNS DAX query.

    Args:
        measures: List of measure names to compute
        group_by: Optional list of columns to group by. Supports formats:
                  - "'Date'[Year]" (proper DAX)
                  - "Date[Year]" (auto-quotes table)
                  - "Date.Year" (dot notation)
        filters: Optional dict of column->values filters

    Returns:
        The DAX query string
    """
    # Normalize and build the column list for grouping
    if group_by:
        normalized_groups = [_normalize_column_reference(col) for col in group_by]
        group_columns = ", ".join(normalized_groups)
    else:
        group_columns = ""

    # Build the measure expressions
    measure_exprs = []
    for measure in measures:
        # Check if measure has table qualification
        if "[" in measure and "]" in measure:
            measure_exprs.append(f'"{measure}", {measure}')
        else:
            # Assume it's just a measure name, wrap in brackets
            measure_exprs.append(f'"{measure}", [{measure}]')

    measures_str = ", ".join(measure_exprs)

    # Build filter expressions with normalized column references
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

    # Build the query
    if group_columns:
        query = f"EVALUATE TOPN({max_rows}, SUMMARIZECOLUMNS({group_columns}, {measures_str}{filter_str}))"
    else:
        query = f"EVALUATE ROW({measures_str})"

    return query


@mcp.tool()
async def list_semantic_models(
    workspace: Optional[str] = None, ctx: Context = None
) -> str:
    """List all semantic models in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace (optional)
        ctx: Context object containing client information

    Returns:
        A string containing the list of semantic models or an error message.
    """
    try:
        # Get workspace from parameter or cache
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        client = SemanticModelClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        models = await client.list_semantic_models(ws)

        if not models:
            return f"No semantic models found in workspace '{ws}'."

        markdown = f"# Semantic Models in workspace '{ws}'\n\n"
        markdown += "| ID | Name | Folder ID | Description |\n"
        markdown += "|-----|------|-----------|-------------|\n"

        for model in models:
            markdown += f"| {model.get('id', 'N/A')} | {model.get('displayName', 'N/A')} | {model.get('folderId', 'N/A')} | {model.get('description', 'N/A')} |\n"

        markdown += f"\n*{len(models)} semantic model(s) found*"

        return markdown

    except Exception as e:
        return f"Error listing semantic models: {str(e)}"


@mcp.tool()
async def get_semantic_model(
    workspace: Optional[str] = None,
    model_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get a specific semantic model by ID.

    Args:
        workspace: Name or ID of the workspace (optional)
        model_id: ID of the semantic model (optional)
        ctx: Context object containing client information

    Returns:
        A string containing the details of the semantic model or an error message.
    """
    try:
        # Get workspace from parameter or cache
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        # Get model_id from parameter or cache
        mid = model_id or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not mid:
            return "Error: No model_id specified and no semantic model context set. Use set_semantic_model first or provide model_id parameter."

        client = SemanticModelClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )

        model = await client.get_semantic_model(ws, mid)

        if not model:
            return f"No semantic model found with ID '{mid}' in workspace '{ws}'."

        return f"Semantic Model '{model.get('displayName', 'Unknown')}' details:\n\n{model}"

    except Exception as e:
        return f"Error retrieving semantic model: {str(e)}"


@mcp.tool()
async def set_semantic_model(
    dataset: str,
    workspace: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Set the current semantic model context for subsequent operations.

    This sets the default semantic model to use for DAX queries, measure evaluation,
    and other operations that require a dataset/semantic model.

    Args:
        dataset: Dataset/semantic model name or ID
        workspace: Workspace name or ID (uses current context if not provided)
        ctx: Context object

    Returns:
        Confirmation message with the set context

    Example:
        set_semantic_model("Sales Model", workspace="Finvision")
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve dataset to ID
        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset, fabric_client)

        # Get dataset name for confirmation
        models = await fabric_client.get_semantic_models(workspace_id)
        model_name = next(
            (m.get("displayName") for m in models if m.get("id") == dataset_id),
            dataset
        )

        # Store in context
        __ctx_cache[f"{ctx.client_id}_semantic_model"] = dataset_id
        __ctx_cache[f"{ctx.client_id}_semantic_model_name"] = model_name

        # Also update workspace context if it was provided
        if workspace:
            __ctx_cache[f"{ctx.client_id}_workspace"] = workspace_id

        return f"✓ Semantic model context set to '{model_name}' (ID: {dataset_id})"

    except Exception as e:
        logger.error(f"Error setting semantic model: {e}")
        return f"Error setting semantic model: {str(e)}"


@mcp.tool()
async def execute_dax_query(
    dax_query: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    max_rows: Optional[int] = 1000,
    ctx: Context = None
) -> str:
    """Execute a DAX query against a Power BI semantic model.

    Args:
        dax_query: The DAX query to execute. Must start with EVALUATE or be a valid DAX expression.
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        max_rows: Maximum rows to return (default 1000, max 100000). If the query doesn't
                  already have TOPN, one will be applied.
        ctx: Context object

    Returns:
        Query results as a formatted markdown table

    Examples:
        execute_dax_query("EVALUATE 'Sales'")
        execute_dax_query("EVALUATE TOPN(10, 'Customers')")
        execute_dax_query("EVALUATE SUMMARIZECOLUMNS('Date'[Year], \"Total\", SUM('Sales'[Amount]))")
        execute_dax_query("EVALUATE ROW(\"Test\", 1 + 1)")

    Note:
        - Requires "Dataset Execute Queries REST API" tenant setting to be enabled
        - Maximum 100,000 rows per query
        - Query timeout is 120 seconds
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve dataset
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set. Use set_semantic_model first or provide dataset parameter."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Validate max_rows
        if max_rows:
            max_rows = min(max(1, max_rows), 100000)

        # Apply TOPN if max_rows specified and not already present
        query = dax_query.strip()
        if max_rows and "TOPN" not in query.upper() and "ROW(" not in query.upper():
            # Wrap query in TOPN
            if query.upper().startswith("EVALUATE"):
                # Extract the expression after EVALUATE
                expr = query[8:].strip()
                query = f"EVALUATE TOPN({max_rows}, {expr})"
            else:
                # Assume it's already an expression, wrap it
                query = f"EVALUATE TOPN({max_rows}, {query})"

        logger.info(f"Executing DAX query on dataset {dataset_id}: {query[:100]}...")

        # Execute the query
        results = await powerbi_client.execute_dax(workspace_id, dataset_id, query)

        # Format and return results
        markdown = _format_dax_results_as_markdown(results)

        return f"## DAX Query Results\n\n**Query:** `{dax_query[:200]}{'...' if len(dax_query) > 200 else ''}`\n\n{markdown}"

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error executing DAX query: {e}")
        return f"Error executing DAX query: {str(e)}"


@mcp.tool()
async def read_semantic_model_table(
    table_name: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    columns: Optional[List[str]] = None,
    max_rows: int = 100,
    ctx: Context = None
) -> str:
    """Read data from a table in a Power BI semantic model.

    Args:
        table_name: Name of the table to read (without quotes)
        workspace: Workspace name or ID
        dataset: Dataset/semantic model name or ID
        columns: Specific columns to return (all if not specified).
                 Format: ["Column1", "Column2"] or ["'Table'[Column1]", "'Table'[Column2]"]
        max_rows: Maximum rows to return (default 100, max 100000)
        ctx: Context object

    Returns:
        Table data as formatted markdown

    Examples:
        read_semantic_model_table("Sales", max_rows=50)
        read_semantic_model_table("Products", columns=["ProductName", "Price"])
        read_semantic_model_table("Customers", columns=["Name", "City", "Country"], max_rows=20)

    Note:
        This generates and executes a DAX EVALUATE query against the table.
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

        # Resolve dataset
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Validate max_rows
        max_rows = min(max(1, max_rows), 100000)

        # Build DAX query
        table_ref = f"'{table_name}'"

        if columns:
            # Build SELECTCOLUMNS expression
            col_exprs = []
            for col in columns:
                if "[" in col:
                    # Already has table/column reference
                    col_exprs.append(f'"{col}", {col}')
                else:
                    # Simple column name - reference from the table
                    col_exprs.append(f'"{col}", {table_ref}[{col}]')

            cols_str = ", ".join(col_exprs)
            dax_query = f"EVALUATE TOPN({max_rows}, SELECTCOLUMNS({table_ref}, {cols_str}))"
        else:
            # Return all columns
            dax_query = f"EVALUATE TOPN({max_rows}, {table_ref})"

        logger.info(f"Reading table {table_name}: {dax_query}")

        # Execute the query
        results = await powerbi_client.execute_dax(workspace_id, dataset_id, dax_query)

        # Format and return results
        markdown = _format_dax_results_as_markdown(results)

        return f"## Table: {table_name}\n\n{markdown}"

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error reading table: {e}")
        return f"Error reading table: {str(e)}"


@mcp.tool()
async def evaluate_measure(
    measures: List[str],
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    group_by: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[str]]] = None,
    max_rows: int = 1000,
    ctx: Context = None
) -> str:
    """Evaluate measures from a semantic model with optional grouping and filters.

    Args:
        measures: List of measure names to compute (e.g., ["Total Sales", "Profit Margin"])
        workspace: Workspace name or ID
        dataset: Dataset/semantic model name or ID
        group_by: Columns to group by. Multiple formats supported:
                  - "'Date'[Year]" (proper DAX format)
                  - "Date[Year]" (auto-adds quotes)
                  - "Date.Year" (dot notation)
        filters: Filter conditions as dict of column->values.
                 Column names support same formats as group_by.
                 Example: {"Asset[AssetCode]": ["BNK001"], "Date[Period]": ["202503"]}
        max_rows: Maximum rows to return (default 1000)
        ctx: Context object

    Returns:
        Computed measures as formatted markdown table

    Examples:
        evaluate_measure(["Total Revenue"])
        evaluate_measure(["Total Revenue"], group_by=["Date[Year]"])
        evaluate_measure(["Net Asset Value"], filters={"Asset[AssetCode]": ["BNK001"], "Date[Period]": ["202503"]})
        evaluate_measure(["Sales", "Profit"], group_by=["Region[Country]"], filters={"Date[Year]": ["2024"]})

    Note:
        - Generates SUMMARIZECOLUMNS or ROW DAX query
        - Filter values should be strings (even for numbers)
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

        # Resolve dataset
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Validate inputs
        if not measures:
            return "Error: At least one measure must be specified."

        max_rows = min(max(1, max_rows), 100000)

        # Build the DAX query
        dax_query = _build_summarize_columns_query(measures, group_by, filters, max_rows)

        logger.info(f"Evaluating measures: {dax_query}")

        # Execute the query
        results = await powerbi_client.execute_dax(workspace_id, dataset_id, dax_query)

        # Format and return results
        markdown = _format_dax_results_as_markdown(results)

        # Build description
        desc = f"**Measures:** {', '.join(measures)}"
        if group_by:
            desc += f"\n**Grouped by:** {', '.join(group_by)}"
        if filters:
            filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items())
            desc += f"\n**Filters:** {filter_desc}"

        return f"## Measure Evaluation\n\n{desc}\n\n{markdown}"

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error evaluating measures: {e}")
        return f"Error evaluating measures: {str(e)}"


@mcp.tool()
async def get_semantic_model_schema(
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get the schema of a semantic model (tables, columns, measures, relationships).

    Uses the Fabric API getDefinition endpoint to retrieve the model definition.

    Args:
        workspace: Workspace name or ID
        dataset: Dataset/semantic model name or ID
        ctx: Context object

    Returns:
        Model schema including:
        - Tables with columns and data types
        - Measures with DAX expressions
        - Relationships between tables

    Note:
        Requires read access to the semantic model definition.
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

        # Resolve dataset
        dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Get model name for display
        if not dataset_name or dataset_name == dataset_id:
            models = await fabric_client.get_semantic_models(workspace_id)
            model_info = next((m for m in models if m.get("id") == dataset_id), {})
            dataset_name = model_info.get("displayName", dataset_id)

        # Call getDefinition endpoint via Fabric API
        logger.info(f"Getting schema for semantic model {dataset_id}")

        definition = await fabric_client._make_request(
            f"workspaces/{workspace_id}/semanticModels/{dataset_id}/getDefinition",
            method="POST",
            lro=True,
            lro_poll_interval=2,
            lro_timeout=120
        )

        if not definition:
            return f"Error: Could not retrieve definition for semantic model '{dataset_name}'"

        # Parse the definition and format as markdown
        markdown = f"# Schema: {dataset_name}\n\n"

        # The definition contains parts with TMDL files
        parts = definition.get("definition", {}).get("parts", [])

        if not parts:
            return f"No schema definition found for '{dataset_name}'"

        # Extract and decode TMDL content
        tables_content = []
        relationships_content = []
        model_content = []

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
            elif "model.tmdl" in path:
                model_content.append(content)

        # Format tables
        if tables_content:
            markdown += "## Tables\n\n"
            for path, content in sorted(tables_content):
                table_name = path.split("/tables/")[-1].replace(".tmdl", "") if "/tables/" in path else "Unknown"
                markdown += f"### {table_name}\n\n```\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```\n\n"

        # Format relationships
        if relationships_content:
            markdown += "## Relationships\n\n"
            for content in relationships_content:
                markdown += f"```\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```\n\n"

        # Summary
        markdown += f"\n*{len(tables_content)} table(s) found*"

        return markdown

    except Exception as e:
        logger.error(f"Error getting semantic model schema: {e}")
        return f"Error getting semantic model schema: {str(e)}"


@mcp.tool()
async def get_semantic_model_measures(
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    search: Optional[str] = None,
    include_expression: bool = True,
    ctx: Context = None
) -> str:
    """List all measures in a semantic model with their DAX expressions.

    This tool is essential for understanding what calculations are available in the model.
    Use this before evaluating measures to find the correct measure names.

    Args:
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        search: Optional search term to filter measures (case-insensitive)
        include_expression: Whether to include the DAX expression (default: True)
        ctx: Context object

    Returns:
        List of measures with:
        - Measure name
        - Table where measure is defined
        - DAX expression
        - Format string (if defined)
        - Description (if defined)
        - Display folder (if defined)

    Examples:
        get_semantic_model_measures()  # List all measures
        get_semantic_model_measures(search="net asset")  # Find measures containing "net asset"
        get_semantic_model_measures(search="revenue", include_expression=False)  # Names only

    Note:
        Use this tool to discover available measures before using evaluate_measure.
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

        # Resolve dataset
        dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Get model name for display
        if not dataset_name or dataset_name == dataset_id:
            models = await fabric_client.get_semantic_models(workspace_id)
            model_info = next((m for m in models if m.get("id") == dataset_id), {})
            dataset_name = model_info.get("displayName", dataset_id)

        logger.info(f"Getting measures for semantic model {dataset_id}")

        # Get TMDL parts
        parts = await _get_model_definition_parts(fabric_client, powerbi_client, workspace_id, dataset_id)

        if not parts:
            return f"Error: Could not retrieve definition for semantic model '{dataset_name}'"

        # Parse measures from all table TMDL files
        all_measures = []

        for path, content in parts:
            if "/tables/" in path and path.endswith(".tmdl"):
                table_name = _parse_tmdl_table_name(content)
                measures = _parse_tmdl_measures(content)

                for measure in measures:
                    measure['table'] = table_name
                    all_measures.append(measure)

        # Filter by search term if provided
        if search:
            search_lower = search.lower()
            all_measures = [
                m for m in all_measures
                if search_lower in m['name'].lower()
                or search_lower in m.get('description', '').lower()
                or search_lower in m.get('displayFolder', '').lower()
            ]

        if not all_measures:
            if search:
                return f"No measures found matching '{search}' in '{dataset_name}'"
            return f"No measures found in semantic model '{dataset_name}'"

        # Sort by table and name
        all_measures.sort(key=lambda m: (m.get('table', ''), m['name']))

        # Format as markdown
        markdown = f"# Measures in {dataset_name}\n\n"

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
                if len(expr) > 500:
                    markdown += f"\n```dax\n{expr[:500]}...\n```\n\n"
                else:
                    markdown += f"\n```dax\n{expr}\n```\n\n"
            else:
                markdown += "\n"

        return markdown

    except Exception as e:
        logger.error(f"Error getting semantic model measures: {e}")
        return f"Error getting semantic model measures: {str(e)}"


@mcp.tool()
async def get_refresh_history(
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    top: int = 10,
    ctx: Context = None
) -> str:
    """Get the refresh history for a semantic model.

    Shows recent refresh operations including their status, timing, and any errors.
    Useful for monitoring refresh health and troubleshooting failed refreshes.

    Args:
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        top: Number of refresh entries to return (default: 10, max: 100)
        ctx: Context object

    Returns:
        Refresh history with:
        - Refresh type (Scheduled, OnDemand, ViaApi, ViaEnhancedApi)
        - Start and end times
        - Status (Completed, Failed, Unknown, Disabled, Cancelled)
        - Duration
        - Error details for failed refreshes

    Examples:
        get_refresh_history()  # Last 10 refreshes for current model
        get_refresh_history(top=25)  # Last 25 refreshes
        get_refresh_history(dataset="Sales Model")  # Specific model

    Note:
        Requires at least read permissions on the semantic model.
    """
    try:
        credential = get_azure_credentials(ctx.client_id, __ctx_cache)
        fabric_client = FabricApiClient(credential)
        powerbi_client = PowerBIClient(credential)

        # Resolve workspace
        workspace_id = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not workspace_id:
            return "Error: No workspace specified and no workspace context set. Use set_workspace first or provide workspace parameter."

        workspace_id = await fabric_client.resolve_workspace(workspace_id)

        # Resolve dataset
        dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set. Use set_semantic_model first or provide dataset parameter."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Get model name for display
        if not dataset_name or dataset_name == dataset_id:
            models = await fabric_client.get_semantic_models(workspace_id)
            model_info = next((m for m in models if m.get("id") == dataset_id), {})
            dataset_name = model_info.get("displayName", dataset_id)

        logger.info(f"Getting refresh history for semantic model {dataset_id}")

        # Get refresh history
        refreshes = await powerbi_client.get_refresh_history(workspace_id, dataset_id, top)

        if not refreshes:
            return f"No refresh history found for semantic model '{dataset_name}'"

        # Format as markdown
        markdown = f"# Refresh History: {dataset_name}\n\n"
        markdown += f"**{len(refreshes)} refresh(es) shown**\n\n"

        markdown += "| Status | Type | Start Time | End Time | Duration |\n"
        markdown += "|--------|------|------------|----------|----------|\n"

        for refresh in refreshes:
            status = refresh.get("status", "Unknown")
            refresh_type = refresh.get("refreshType", "Unknown")
            start_time = refresh.get("startTime", "N/A")
            end_time = refresh.get("endTime", "N/A")

            # Calculate duration if both times are available
            duration = "N/A"
            if start_time != "N/A" and end_time != "N/A":
                try:
                    from datetime import datetime
                    # Parse ISO 8601 format
                    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                    delta = end_dt - start_dt
                    total_seconds = int(delta.total_seconds())
                    if total_seconds < 60:
                        duration = f"{total_seconds}s"
                    elif total_seconds < 3600:
                        minutes = total_seconds // 60
                        seconds = total_seconds % 60
                        duration = f"{minutes}m {seconds}s"
                    else:
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        duration = f"{hours}h {minutes}m"
                except Exception:
                    duration = "N/A"

            # Format timestamps for readability
            if start_time != "N/A":
                start_time = start_time.replace("T", " ").replace("Z", " UTC")[:22]
            if end_time != "N/A":
                end_time = end_time.replace("T", " ").replace("Z", " UTC")[:22]

            # Status emoji
            status_icon = {
                "Completed": "✓",
                "Failed": "✗",
                "Unknown": "?",
                "Disabled": "⊘",
                "Cancelled": "⊗"
            }.get(status, "")

            markdown += f"| {status_icon} {status} | {refresh_type} | {start_time} | {end_time} | {duration} |\n"

        # Add error details for failed refreshes
        failed_refreshes = [r for r in refreshes if r.get("status") == "Failed"]
        if failed_refreshes:
            markdown += "\n## Failed Refresh Details\n\n"
            for refresh in failed_refreshes:
                start_time = refresh.get("startTime", "Unknown time")
                if start_time != "Unknown time":
                    start_time = start_time.replace("T", " ").replace("Z", " UTC")[:22]

                markdown += f"### Refresh at {start_time}\n\n"

                # Parse error JSON if available
                error_json = refresh.get("serviceExceptionJson")
                if error_json:
                    try:
                        import json
                        error_data = json.loads(error_json)
                        error_code = error_data.get("errorCode", "Unknown")
                        error_desc = error_data.get("errorDescription", "No description available")
                        markdown += f"**Error Code:** {error_code}\n\n"
                        markdown += f"**Description:** {error_desc}\n\n"
                    except Exception:
                        markdown += f"**Error:** {error_json}\n\n"
                else:
                    markdown += "*No error details available*\n\n"

        return markdown

    except Exception as e:
        logger.error(f"Error getting refresh history: {e}")
        return f"Error getting refresh history: {str(e)}"


@mcp.tool()
async def get_semantic_model_tables(
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    include_hidden: bool = False,
    ctx: Context = None
) -> str:
    """List all tables and columns in a semantic model.

    This tool helps you understand the model structure for building filters
    and group-by clauses when evaluating measures.

    Args:
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        include_hidden: Whether to include hidden columns (default: False)
        ctx: Context object

    Returns:
        List of tables with:
        - Table name
        - Columns with data types
        - Which columns can be used for filtering/grouping

    Examples:
        get_semantic_model_tables()  # List all tables and columns
        get_semantic_model_tables(include_hidden=True)  # Include hidden columns

    Note:
        Use this tool to find column names for filters and group_by parameters
        in the evaluate_measure tool. Column references should use the format:
        'TableName'[ColumnName]
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

        # Resolve dataset
        dataset_name = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model_name")
        dataset_id = dataset or __ctx_cache.get(f"{ctx.client_id}_semantic_model")
        if not dataset_id:
            return "Error: No dataset specified and no semantic model context set."

        dataset_id = await powerbi_client.resolve_dataset_id(workspace_id, dataset_id, fabric_client)

        # Get model name for display
        if not dataset_name or dataset_name == dataset_id:
            models = await fabric_client.get_semantic_models(workspace_id)
            model_info = next((m for m in models if m.get("id") == dataset_id), {})
            dataset_name = model_info.get("displayName", dataset_id)

        logger.info(f"Getting tables for semantic model {dataset_id}")

        # Get TMDL parts
        parts = await _get_model_definition_parts(fabric_client, powerbi_client, workspace_id, dataset_id)

        if not parts:
            return f"Error: Could not retrieve definition for semantic model '{dataset_name}'"

        # Parse tables and columns from TMDL files
        all_tables = []

        for path, content in parts:
            if "/tables/" in path and path.endswith(".tmdl"):
                table_name = _parse_tmdl_table_name(content)
                columns = _parse_tmdl_columns(content)

                # Filter hidden columns unless requested
                if not include_hidden:
                    columns = [c for c in columns if not c.get('isHidden', False)]

                all_tables.append({
                    'name': table_name,
                    'columns': columns
                })

        if not all_tables:
            return f"No tables found in semantic model '{dataset_name}'"

        # Sort tables by name
        all_tables.sort(key=lambda t: t['name'])

        # Format as markdown
        markdown = f"# Tables in {dataset_name}\n\n"
        markdown += f"**{len(all_tables)} table(s) found**\n\n"
        markdown += "Use column references in format: `'TableName'[ColumnName]`\n\n"

        for table in all_tables:
            table_name = table['name']
            columns = table['columns']

            markdown += f"## {table_name}\n\n"

            if not columns:
                markdown += "*No visible columns*\n\n"
                continue

            markdown += "| Column | Data Type | Description |\n"
            markdown += "|--------|-----------|-------------|\n"

            for col in columns:
                col_name = col['name']
                data_type = col.get('dataType', 'unknown')
                description = col.get('description', '')

                # Show DAX reference format
                col_ref = f"`'{table_name}'[{col_name}]`"

                markdown += f"| {col_ref} | {data_type} | {description} |\n"

            markdown += "\n"

        return markdown

    except Exception as e:
        logger.error(f"Error getting semantic model tables: {e}")
        return f"Error getting semantic model tables: {str(e)}"
