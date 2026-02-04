# Implementation Plan: Dataset & Report Querying for MCP

## Overview

Add querying capabilities to the MCP server for Power BI semantic models and reports. This enables users to execute DAX queries, read table data, compute measures, and interact with reports programmatically.

**Status:** Ready for Implementation
**Validated:** Tenant setting "Dataset Execute Queries REST API" is enabled ✅
**Test Date:** 2026-02-04

---

## Architecture

### API Endpoints Used

| Endpoint | Purpose | Base URL |
|----------|---------|----------|
| `POST /datasets/{id}/executeQueries` | Execute DAX queries | `api.powerbi.com` |
| `GET /reports/{id}/pages` | List report pages | `api.powerbi.com` |
| `POST /reports/{id}/ExportTo` | Export report to file | `api.powerbi.com` |
| `GET /reports/{id}/exports/{exportId}` | Check export status | `api.powerbi.com` |

### Authentication

- **Scope:** `https://analysis.windows.net/powerbi/api/.default`
- **Note:** Different from Fabric API scope (`api.fabric.microsoft.com`)
- Uses existing `get_azure_credentials()` from authentication helper

---

## Phase 1: Core DAX Querying (High Priority)

### 1.1 New Client: `powerbi_client.py`

**Location:** `helpers/clients/powerbi_client.py`

```python
class PowerBIClient:
    """Client for Power BI REST API (separate from Fabric API)."""

    BASE_URL = "https://api.powerbi.com/v1.0/myorg"
    SCOPE = "https://analysis.windows.net/powerbi/api/.default"

    def __init__(self, credential):
        self.credential = credential

    async def execute_dax(self, workspace_id, dataset_id, dax_query, include_nulls=True):
        """Execute a DAX query against a semantic model."""

    async def get_report_pages(self, workspace_id, report_id):
        """Get all pages in a report."""

    async def export_report(self, workspace_id, report_id, format, ...):
        """Start report export job."""

    async def get_export_status(self, workspace_id, report_id, export_id):
        """Check export job status."""

    async def download_export(self, workspace_id, report_id, export_id):
        """Download completed export file."""
```

### 1.2 New Tool: `execute_dax_query`

**Location:** `tools/semantic_model.py`

```python
@mcp.tool()
async def execute_dax_query(
    dax_query: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    max_rows: Optional[int] = 1000,
    ctx: Context = None
) -> str:
    """
    Execute a DAX query against a Power BI semantic model.

    Args:
        dax_query: The DAX query to execute (e.g., "EVALUATE 'TableName'")
        workspace: Workspace name or ID (uses context if not provided)
        dataset: Dataset/semantic model name or ID (uses context if not provided)
        max_rows: Maximum rows to return (default 1000, max 100000)

    Returns:
        Query results as a formatted markdown table

    Example:
        execute_dax_query("EVALUATE TOPN(10, 'Sales')")
        execute_dax_query("EVALUATE SUMMARIZECOLUMNS('Date'[Year], \"Total\", SUM('Sales'[Amount]))")
    """
```

**Implementation Details:**
- Wrap query with `TOPN()` if max_rows specified and not already present
- Format results as markdown table
- Handle errors gracefully with clear messages
- Support both workspace/dataset names and IDs

### 1.3 New Tool: `read_semantic_model_table`

**Location:** `tools/semantic_model.py`

```python
@mcp.tool()
async def read_semantic_model_table(
    table_name: str,
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    columns: Optional[List[str]] = None,
    max_rows: int = 100,
    ctx: Context = None
) -> str:
    """
    Read data from a table in a Power BI semantic model.

    Args:
        table_name: Name of the table to read
        workspace: Workspace name or ID
        dataset: Dataset/semantic model name or ID
        columns: Specific columns to return (all if not specified)
        max_rows: Maximum rows to return (default 100)

    Returns:
        Table data as formatted markdown

    Example:
        read_semantic_model_table("Sales", max_rows=50)
        read_semantic_model_table("Products", columns=["ProductName", "Price"])
    """
```

**Implementation Details:**
- Generates DAX: `EVALUATE TOPN({max_rows}, SELECTCOLUMNS('{table}', ...))`
- If no columns specified: `EVALUATE TOPN({max_rows}, '{table}')`

### 1.4 New Tool: `evaluate_measure`

**Location:** `tools/semantic_model.py`

```python
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
    """
    Evaluate measures from a semantic model with optional grouping and filters.

    Args:
        measures: List of measure names to compute (e.g., ["Total Sales", "Profit Margin"])
        workspace: Workspace name or ID
        dataset: Dataset/semantic model name or ID
        group_by: Columns to group by (e.g., ["'Date'[Year]", "'Product'[Category]"])
        filters: Filter conditions (e.g., {"'Date'[Year]": ["2023", "2024"]})
        max_rows: Maximum rows to return

    Returns:
        Computed measures as formatted markdown table

    Example:
        evaluate_measure(["Total Revenue"], group_by=["'Date'[Year]"])
        evaluate_measure(["Sales", "Profit"], group_by=["'Region'[Country]"], filters={"'Date'[Year]": ["2024"]})
    """
```

**Implementation Details:**
- Generates DAX using `SUMMARIZECOLUMNS()` or `CALCULATETABLE()`
- Properly escapes measure and column names
- Applies `FILTER()` for filter conditions

---

## Phase 2: Report Querying (Medium Priority)

### 2.1 New Tool: `get_report_pages`

**Location:** `tools/report.py`

```python
@mcp.tool()
async def get_report_pages(
    workspace: Optional[str] = None,
    report_id: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    List all pages in a Power BI report.

    Args:
        workspace: Workspace name or ID
        report_id: Report ID

    Returns:
        List of report pages with names and display names
    """
```

**API:** `GET /reports/{reportId}/pages`

### 2.2 New Tool: `export_report`

**Location:** `tools/report.py`

```python
@mcp.tool()
async def export_report(
    workspace: Optional[str] = None,
    report_id: Optional[str] = None,
    format: str = "PDF",
    page_name: Optional[str] = None,
    visual_name: Optional[str] = None,
    bookmark_name: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Export a Power BI report to a file (PDF, PPTX, or PNG).

    Args:
        workspace: Workspace name or ID
        report_id: Report ID
        format: Export format - "PDF", "PPTX", or "PNG"
        page_name: Specific page to export (optional)
        visual_name: Specific visual to export (optional, requires page_name)
        bookmark_name: Bookmark to apply before export (optional)

    Returns:
        Export status and download URL when complete

    Note:
        Requires Premium, Embedded, or Fabric capacity.
        Export is asynchronous - tool will poll until complete.
    """
```

**Implementation Details:**
- Starts async export job via `POST /reports/{id}/ExportTo`
- Polls status via `GET /reports/{id}/exports/{exportId}`
- Returns download URL when complete
- Timeout after configurable period (default 5 minutes)

### 2.3 New Tool: `get_report_definition`

**Location:** `tools/report.py`

```python
@mcp.tool()
async def get_report_definition(
    workspace: Optional[str] = None,
    report_id: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Get the structure/definition of a Power BI report.

    Returns information about:
    - Pages and their layout
    - Visuals on each page
    - Filters (report, page, visual level)
    - Data sources used

    Note: Uses Fabric API getDefinition endpoint.
    """
```

**API:** `POST /reports/{reportId}/getDefinition` (Fabric API)

---

## Phase 3: Model Metadata (Lower Priority)

### 3.1 New Tool: `get_semantic_model_schema`

**Location:** `tools/semantic_model.py`

```python
@mcp.tool()
async def get_semantic_model_schema(
    workspace: Optional[str] = None,
    dataset: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Get the schema of a semantic model (tables, columns, measures, relationships).

    Returns:
        Model schema including:
        - Tables with columns and data types
        - Measures with DAX expressions
        - Relationships between tables

    Note: Requires XMLA read access or uses Fabric getDefinition API.
    """
```

**Implementation Options:**
1. **Fabric API:** `POST /semanticModels/{id}/getDefinition` - returns TMDL/TMSL
2. **XMLA Endpoint:** DMV queries (requires Premium + XMLA enabled)
3. **Discover via DAX:** Limited - can probe for known table names

### 3.2 New Tool: `set_semantic_model`

**Location:** `tools/semantic_model.py`

```python
@mcp.tool()
async def set_semantic_model(
    dataset: str,
    workspace: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Set the current semantic model context for subsequent operations.

    Args:
        dataset: Dataset/semantic model name or ID
        workspace: Workspace name or ID (uses current context if not provided)
    """
```

---

## File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `helpers/clients/powerbi_client.py` | Power BI REST API client |

### Modified Files

| File | Changes |
|------|---------|
| `helpers/clients/__init__.py` | Export `PowerBIClient` |
| `tools/semantic_model.py` | Add 4 new tools |
| `tools/report.py` | Add 3 new tools |
| `helpers/utils/authentication.py` | Add Power BI scope constant |

---

## API Constraints & Limits

| Constraint | Limit |
|------------|-------|
| Max rows per query | 100,000 |
| Max values per query | 1,000,000 |
| Max data size per query | 15 MB |
| Rate limit | 120 requests/min/user |
| Export timeout | ~5 minutes typical |

### Not Supported via REST API

- MDX queries
- INFO functions (e.g., `INFO.TABLES()`)
- DMV queries
- Queries on SSO-enabled datasets
- Queries requiring RLS with Service Principal

---

## Implementation Order

```
Week 1: Phase 1 - Core DAX Querying
├── [ ] Create PowerBIClient class
├── [ ] Implement execute_dax_query tool
├── [ ] Implement read_semantic_model_table tool
├── [ ] Implement evaluate_measure tool
├── [ ] Add unit tests
└── [ ] Test with real datasets

Week 2: Phase 2 - Report Querying
├── [ ] Implement get_report_pages tool
├── [ ] Implement export_report tool
├── [ ] Implement get_report_definition tool
├── [ ] Add unit tests
└── [ ] Test with real reports

Week 3: Phase 3 - Model Metadata (Optional)
├── [ ] Implement get_semantic_model_schema tool
├── [ ] Implement set_semantic_model tool
├── [ ] Investigate XMLA endpoint access
└── [ ] Documentation updates
```

---

## Testing Checklist

### Phase 1 Tests

- [ ] Execute simple DAX query (`EVALUATE ROW("Test", 1)`)
- [ ] Execute query with TOPN limit
- [ ] Execute SUMMARIZECOLUMNS query
- [ ] Read table data with column selection
- [ ] Evaluate measure with grouping
- [ ] Evaluate measure with filters
- [ ] Handle dataset not found error
- [ ] Handle invalid DAX syntax error
- [ ] Handle rate limiting (429)
- [ ] Verify max_rows enforcement

### Phase 2 Tests

- [ ] List pages of a report
- [ ] Export report to PDF
- [ ] Export specific page to PNG
- [ ] Export specific visual
- [ ] Handle export timeout
- [ ] Handle capacity requirement error
- [ ] Get report definition

---

## Example Usage

### Execute DAX Query

```
User: Run a DAX query to get top 10 customers by sales

Tool call: execute_dax_query(
    dax_query="EVALUATE TOPN(10, SUMMARIZECOLUMNS('Customer'[Name], \"TotalSales\", SUM('Sales'[Amount])), [TotalSales], DESC)",
    workspace="Finvision",
    dataset="Sales Model"
)
```

### Read Table

```
User: Show me the first 20 products

Tool call: read_semantic_model_table(
    table_name="Products",
    max_rows=20,
    workspace="Finvision",
    dataset="Sales Model"
)
```

### Evaluate Measure

```
User: What's the total revenue by year?

Tool call: evaluate_measure(
    measures=["Total Revenue"],
    group_by=["'Date'[Year]"],
    workspace="Finvision",
    dataset="Sales Model"
)
```

### Export Report

```
User: Export the sales dashboard to PDF

Tool call: export_report(
    report_id="abc-123",
    format="PDF",
    workspace="Finvision"
)
```

---

## Dependencies

No new dependencies required. Uses existing:
- `requests` - HTTP client
- `azure-identity` - Authentication
- `cachetools` - Token caching

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Tenant setting disabled | Test script validates setting; clear error message |
| Rate limiting | Implement exponential backoff; respect Retry-After header |
| Large result sets | Enforce max_rows; warn user about truncation |
| Export capacity requirement | Check capacity before export; clear error for Pro-only workspaces |
| Token scope issues | Separate Power BI scope from Fabric scope |

---

## Success Criteria

1. ✅ Can execute any valid DAX query against any accessible semantic model
2. ✅ Can read table data with optional column filtering
3. ✅ Can evaluate measures with grouping and filters
4. ✅ Can list report pages
5. ✅ Can export reports to PDF/PPTX/PNG (on Premium/Fabric capacity)
6. ✅ Clear error messages for all failure modes
7. ✅ Results formatted as readable markdown tables
