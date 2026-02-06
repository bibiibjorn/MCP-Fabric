from helpers.utils.context import mcp, __ctx_cache
from mcp.server.fastmcp import Context
from helpers.utils.authentication import get_azure_credentials
from helpers.clients import (
    FabricApiClient,
    NotebookClient,
)
from helpers.clients.spark_client import SparkClient
from helpers.utils import _is_valid_uuid
import json
from helpers.logging_config import get_logger
from datetime import datetime


from typing import Optional, Dict, List, Any
import base64
import re

logger = get_logger(__name__)


@mcp.tool()
async def manage_notebook(
    action: str,
    workspace: Optional[str] = None,
    notebook_name: Optional[str] = None,
    notebook_id: Optional[str] = None,
    template_type: Optional[str] = "basic",
    cell_index: Optional[int] = None,
    cell_content: Optional[str] = None,
    cell_type: Optional[str] = "code",
    code_type: Optional[str] = None,
    operation: Optional[str] = None,
    source_table: Optional[str] = None,
    target_table: Optional[str] = None,
    columns: Optional[str] = None,
    filter_condition: Optional[str] = None,
    lakehouse_name: Optional[str] = None,
    code: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Manage Fabric notebooks: list, create, read content, update cells, generate/validate code, and analyze performance.

    Args:
        action: Action to perform. One of:
            - 'list': List all notebooks in a workspace
            - 'create': Create a new notebook from a template
            - 'get_content': Get the content of a specific notebook
            - 'update_cell': Update a specific cell in a notebook
            - 'generate_code': Generate PySpark or Fabric-specific code
            - 'validate_code': Validate PySpark or Fabric code
            - 'analyze_performance': Analyze a notebook for performance optimization
        workspace: Name or ID of the workspace (optional for 'list', required for most actions)
        notebook_name: Name of the notebook (required for 'create')
        notebook_id: ID or name of the notebook (required for 'get_content', 'update_cell', 'analyze_performance')
        template_type: Template type for 'create': 'basic', 'etl', 'analytics', 'ml', 'fabric_integration', 'streaming' (default: 'basic')
        cell_index: Index of the cell to update, 0-based (required for 'update_cell')
        cell_content: New content for the cell (required for 'update_cell')
        cell_type: Type of cell for 'update_cell': 'code' or 'markdown' (default: 'code')
        code_type: Type of code for 'generate_code'/'validate_code': 'pyspark' or 'fabric'
        operation: Operation type for 'generate_code'.
            PySpark ops: 'read_table', 'write_table', 'transform', 'join', 'aggregate', 'schema_inference', 'data_quality', 'performance_optimization'
            Fabric ops: 'read_lakehouse', 'write_lakehouse', 'merge_delta', 'performance_monitor'
        source_table: Source table name for 'generate_code' (format: lakehouse.table_name)
        target_table: Target table name for 'generate_code' (format: lakehouse.table_name)
        columns: Comma-separated list of columns for 'generate_code' (PySpark only)
        filter_condition: Filter condition for 'generate_code' (PySpark only)
        lakehouse_name: Name of the lakehouse for 'generate_code' (Fabric only)
        code: Code to validate for 'validate_code'
        ctx: Context object containing client information

    Returns:
        A string containing the result of the action or an error message.
    """

    # ── action: list ──────────────────────────────────────────────────
    if action == "list":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")

            fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            notebook_client = NotebookClient(fabric_client)

            # Resolve workspace name to ID if needed
            workspace_id = await fabric_client.resolve_workspace(workspace)

            return await notebook_client.list_notebooks(workspace_id)
        except Exception as e:
            logger.error(f"Error listing notebooks: {str(e)}")
            return f"Error listing notebooks: {str(e)}"

    # ── action: create ────────────────────────────────────────────────
    elif action == "create":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")
            if not notebook_name:
                return "Error: 'notebook_name' is required for 'create' action."
            if not workspace:
                return "Error: 'workspace' is required for 'create' action."

            # Fabric-specific templates use the helpers module
            fabric_templates = {"fabric_integration", "streaming"}

            if template_type in fabric_templates:
                from helpers.pyspark_helpers import create_notebook_from_template
                notebook_data = create_notebook_from_template(template_type)
            else:
                # PySpark templates defined inline
                notebook_data = _get_pyspark_template(template_type)
                if notebook_data is None:
                    all_templates = "basic, etl, analytics, ml, fabric_integration, streaming"
                    return f"Invalid template type '{template_type}'. Available templates: {all_templates}"

            notebook_content = json.dumps(notebook_data, indent=2)

            notebook_client = NotebookClient(
                FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            )
            response = await notebook_client.create_notebook(
                workspace, notebook_name, notebook_content
            )

            if isinstance(response, dict) and response.get("id"):
                return f"Created notebook '{notebook_name}' with ID: {response['id']} using '{template_type}' template"
            else:
                return f"Failed to create notebook: {response}"

        except Exception as e:
            logger.error(f"Error creating notebook: {str(e)}")
            return f"Error creating notebook: {str(e)}"

    # ── action: get_content ───────────────────────────────────────────
    elif action == "get_content":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")
            if not notebook_id:
                return "Error: 'notebook_id' is required for 'get_content' action."
            if not workspace:
                return "Error: 'workspace' is required for 'get_content' action."

            fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            notebook_client = NotebookClient(fabric_client)

            # Resolve workspace ID if name provided
            workspace_id = await fabric_client.resolve_workspace(workspace)

            # Resolve notebook ID if name provided
            if not _is_valid_uuid(notebook_id):
                notebook_resolved_id = await fabric_client.resolve_item_id(
                    item=notebook_id, type="Notebook", workspace=workspace_id
                )
            else:
                notebook_resolved_id = notebook_id

            # Get the notebook definition (actual content) using getDefinition endpoint
            definition = await notebook_client.get_notebook_definition(workspace_id, notebook_resolved_id)

            if isinstance(definition, str):  # Error message
                return definition

            # Extract and decode the notebook content from the definition
            parts = definition.get("definition", {}).get("parts", [])

            # Priority order: .ipynb first, then notebook-content.py (Fabric native format)
            for part in parts:
                path = part.get("path", "")
                if path.endswith(".ipynb"):
                    payload = part.get("payload", "")
                    if payload:
                        # Decode base64 content
                        decoded_content = base64.b64decode(payload).decode("utf-8")
                        return decoded_content

            # Fallback to Fabric's native .py format (notebook-content.py)
            for part in parts:
                path = part.get("path", "")
                if path == "notebook-content.py" or path.endswith(".py"):
                    payload = part.get("payload", "")
                    if payload:
                        # Decode base64 content
                        decoded_content = base64.b64decode(payload).decode("utf-8")
                        return decoded_content

            # If no content found, list available parts for debugging
            if parts:
                part_paths = [p.get("path", "unknown") for p in parts]
                return f"No notebook content found. Available parts: {part_paths}"

            return "No notebook content found in the definition."

        except Exception as e:
            logger.error(f"Error getting notebook content: {str(e)}")
            return f"Error getting notebook content: {str(e)}"

    # ── action: update_cell ───────────────────────────────────────────
    elif action == "update_cell":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")
            if not notebook_id:
                return "Error: 'notebook_id' is required for 'update_cell' action."
            if not workspace:
                return "Error: 'workspace' is required for 'update_cell' action."
            if cell_index is None:
                return "Error: 'cell_index' is required for 'update_cell' action."
            if not cell_content:
                return "Error: 'cell_content' is required for 'update_cell' action."

            fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            notebook_client = NotebookClient(fabric_client)

            # Resolve workspace ID
            workspace_id = await fabric_client.resolve_workspace(workspace)

            # Resolve notebook ID and get notebook name
            if not _is_valid_uuid(notebook_id):
                nb_name = notebook_id
                notebook_resolved_id = await fabric_client.resolve_item_id(
                    item=notebook_id, type="Notebook", workspace=workspace_id
                )
            else:
                notebook_resolved_id = notebook_id
                # Get notebook metadata to get the name
                notebook_meta = await notebook_client.get_notebook(workspace_id, notebook_resolved_id)
                if isinstance(notebook_meta, dict):
                    nb_name = notebook_meta.get("displayName", "notebook")
                else:
                    nb_name = "notebook"

            # Get the raw notebook definition to determine format
            definition = await notebook_client.get_notebook_definition(workspace_id, notebook_resolved_id)

            if isinstance(definition, str):
                return f"Error getting notebook definition: {definition}"

            if definition is None:
                return "Error: Failed to get notebook definition (API returned None)"

            # Extract parts from the definition
            parts = definition.get("definition", {}).get("parts", [])

            if not parts:
                return "Error: Notebook definition has no parts"

            # Determine the notebook format and find the content
            notebook_format = None
            content_part = None
            content_path = None

            for part in parts:
                path = part.get("path", "")
                if path.endswith(".ipynb"):
                    notebook_format = "ipynb"
                    content_part = part
                    content_path = path
                    break
                elif path == "notebook-content.py" or path.endswith(".py"):
                    notebook_format = "py"
                    content_part = part
                    content_path = path

            if content_part is None:
                part_paths = [p.get("path", "unknown") for p in parts]
                return f"Error: No notebook content found. Available parts: {part_paths}"

            # Decode the content
            payload = content_part.get("payload", "")
            if not payload:
                return "Error: Notebook content payload is empty"

            try:
                decoded_content = base64.b64decode(payload).decode("utf-8")
            except Exception as e:
                return f"Error decoding notebook content: {str(e)}"

            # Parse content based on format
            if notebook_format == "ipynb":
                try:
                    notebook_data = json.loads(decoded_content)
                    cells = notebook_data.get("cells", [])
                except json.JSONDecodeError as e:
                    return f"Error parsing notebook JSON: {str(e)}"
            else:
                # Parse Fabric's native .py format
                cells = _parse_fabric_py_to_cells(decoded_content)
                # Create a minimal notebook structure
                notebook_data = {
                    "nbformat": 4,
                    "nbformat_minor": 5,
                    "metadata": {"language_info": {"name": "python"}},
                    "cells": cells
                }

            if not cells:
                return "Error: Notebook has no cells"

            if cell_index < 0 or cell_index >= len(cells):
                return f"Cell index {cell_index} is out of range. Notebook has {len(cells)} cells (0-{len(cells)-1})."

            # Prepare source as list of lines (ipynb format)
            if isinstance(cell_content, str):
                source_lines = cell_content.split('\n')
                # Add newline to all lines except the last
                source_lines = [line + '\n' if i < len(source_lines) - 1 else line
                               for i, line in enumerate(source_lines)]
            else:
                source_lines = cell_content

            # Update the cell, preserving existing metadata where possible
            existing_cell = cells[cell_index]
            cells[cell_index] = {
                "cell_type": cell_type,
                "source": source_lines,
                "metadata": existing_cell.get("metadata", {}),
            }

            # Add code-specific fields
            if cell_type == "code":
                cells[cell_index]["execution_count"] = None
                cells[cell_index]["outputs"] = []

            # Update notebook data
            notebook_data["cells"] = cells

            # Serialize back to the original format
            # Always use ipynb format for updateDefinition as it's more reliable
            updated_content = json.dumps(notebook_data, indent=2)

            # Call the update API
            result = await notebook_client.update_notebook_definition(
                workspace=workspace_id,
                notebook_id=notebook_resolved_id,
                content=updated_content,
                notebook_name=nb_name,
            )

            # Check for errors in the result
            if result is None:
                return "Error: Update API returned None - the operation may have failed"

            if isinstance(result, dict) and "error" in result:
                return f"Error updating notebook: {result['error']}"

            return f"Successfully updated cell {cell_index} in notebook '{nb_name}' with {cell_type} content ({len(cell_content)} characters)."

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing notebook JSON: {str(e)}")
            return f"Error parsing notebook content: {str(e)}"
        except Exception as e:
            logger.error(f"Error updating notebook cell: {str(e)}")
            return f"Error updating notebook cell: {str(e)}"

    # ── action: generate_code ─────────────────────────────────────────
    elif action == "generate_code":
        try:
            if not code_type:
                return "Error: 'code_type' is required for 'generate_code' action. Use 'pyspark' or 'fabric'."
            if not operation:
                return "Error: 'operation' is required for 'generate_code' action."

            if code_type == "fabric":
                # Fabric-specific code generation
                from helpers.pyspark_helpers import PySparkCodeGenerator
                generator = PySparkCodeGenerator()

                if operation == "read_lakehouse":
                    if not lakehouse_name or not source_table:
                        return "Error: lakehouse_name and source_table are required for read_lakehouse operation"
                    code_result = generator.generate_fabric_lakehouse_reader(lakehouse_name, source_table)
                elif operation == "write_lakehouse":
                    if not target_table:
                        return "Error: target_table is required for write_lakehouse operation"
                    code_result = generator.generate_fabric_lakehouse_writer(target_table)
                elif operation == "merge_delta":
                    if not target_table:
                        return "Error: target_table is required for merge_delta operation"
                    code_result = generator.generate_delta_merge_operation(target_table, "new_df", "target.id = source.id")
                elif operation == "performance_monitor":
                    code_result = generator.generate_performance_monitoring()
                else:
                    available_ops = "read_lakehouse, write_lakehouse, merge_delta, performance_monitor"
                    return f"Invalid Fabric operation. Available operations: {available_ops}"

                return f"""```python\n{code_result}\n```\n\n**Generated Fabric-specific PySpark code for '{operation}' operation**"""

            else:
                # PySpark code generation
                code_templates = {
                    "read_table": f"""# Read data from table\ndf = spark.table("{source_table or 'lakehouse.table_name'}")\ndf.show()\ndf.printSchema()""",
                    "write_table": f"""# Write data to table\ndf.write \\\n    .format("delta") \\\n    .mode("overwrite") \\\n    .saveAsTable("{target_table or 'lakehouse.output_table'}")\n\nprint(f"Successfully wrote {{df.count()}} records to {target_table or 'lakehouse.output_table'}")""",
                    "transform": f"""# Data transformation\nfrom pyspark.sql.functions import *\n\ndf_transformed = df \\\n    .select({columns or '*'}) \\\n    {f'.filter({filter_condition})' if filter_condition else ''} \\\n    .withColumn("processed_date", current_timestamp())\n\ndf_transformed.show()""",
                    "join": f"""# Join tables\ndf1 = spark.table("{source_table or 'lakehouse.table1'}")\ndf2 = spark.table("{target_table or 'lakehouse.table2'}")\n\n# Inner join (modify join condition as needed)\ndf_joined = df1.join(df2, df1.id == df2.id, "inner")\n\ndf_joined.show()""",
                    "aggregate": f"""# Data aggregation\nfrom pyspark.sql.functions import *\n\ndf_agg = df \\\n    .groupBy({columns or '"column1"'}) \\\n    .agg(\n        count("*").alias("count"),\n        sum("amount").alias("total_amount"),\n        avg("amount").alias("avg_amount"),\n        max("date").alias("max_date")\n    ) \\\n    .orderBy(desc("total_amount"))\n\ndf_agg.show()""",
                    "schema_inference": """# Schema inference and data profiling\nprint("=== Schema Information ===")\ndf.printSchema()\n\nprint("\\n=== Data Profile ===")\nprint(f"Record count: {df.count()}")\nprint(f"Column count: {len(df.columns)}")\n\nprint("\\n=== Column Statistics ===")\ndf.describe().show()""",
                    "data_quality": """# Data quality checks\nfrom pyspark.sql.functions import *\n\nprint("=== Data Quality Report ===")\n\n# Check for duplicates\nduplicate_count = df.count() - df.distinct().count()\nprint(f"Duplicate rows: {duplicate_count}")\n\n# Check for null values\ntotal_rows = df.count()\nfor column in df.columns:\n    null_count = df.filter(col(column).isNull()).count()\n    null_percentage = (null_count / total_rows) * 100\n    print(f"{column}: {null_count} nulls ({null_percentage:.2f}%)")""",
                    "performance_optimization": f"""# Performance optimization techniques\n\n# 1. Cache frequently used DataFrames\ndf.cache()\nprint(f"Cached DataFrame with {{df.count()}} records")\n\n# 2. Repartition for better parallelism\noptimal_partitions = spark.sparkContext.defaultParallelism * 2\ndf_repartitioned = df.repartition(optimal_partitions)\n\n# 3. Use broadcast for small dimension tables (< 200MB)\nfrom pyspark.sql.functions import broadcast\n# df_joined = large_df.join(broadcast(small_df), "key")\n\n# 4. Optimize file formats - use Delta Lake\ndf.write \\\n    .format("delta") \\\n    .mode("overwrite") \\\n    .option("optimizeWrite", "true") \\\n    .option("autoOptimize", "true") \\\n    .saveAsTable("{target_table or 'lakehouse.optimized_table'}")\n\n# 5. Show execution plan\ndf.explain(True)""",
                }

                if operation not in code_templates:
                    available_ops = ", ".join(code_templates.keys())
                    return f"Invalid PySpark operation. Available operations: {available_ops}"

                generated_code = code_templates[operation]

                return f"""```python\n{generated_code}\n```\n\n**Generated PySpark code for '{operation}' operation**\n\nRemember to replace placeholder table names and adjust column names as needed."""

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return f"Error generating code: {str(e)}"

    # ── action: validate_code ─────────────────────────────────────────
    elif action == "validate_code":
        try:
            if not code:
                return "Error: 'code' is required for 'validate_code' action."

            validation_code_type = code_type or "pyspark"

            validation_results = []
            warnings = []
            suggestions = []

            # Basic syntax validation (both types)
            try:
                compile(code, '<string>', 'exec')
                validation_results.append("Syntax validation: PASSED")
            except SyntaxError as e:
                return f"Syntax validation: FAILED - {e}"

            if validation_code_type == "fabric":
                # Fabric-specific validation
                from helpers.pyspark_helpers import PySparkValidator
                validator = PySparkValidator()

                fabric_results = validator.validate_fabric_compatibility(code)
                performance_results = validator.check_performance_patterns(code)

                if 'spark.table(' in code:
                    validation_results.append("Using Fabric managed tables")
                if 'notebookutils' in code:
                    validation_results.append("Using Fabric notebook utilities")
                if 'format("delta")' in code:
                    validation_results.append("Using Delta Lake format")

                if 'spark.sql("USE' in code:
                    warnings.append("Explicit USE statements may not be necessary in Fabric")
                if 'hdfs://' in code or 's3://' in code:
                    warnings.append("Direct file system paths detected - consider using Fabric's managed storage")

                warnings.extend(performance_results.get("warnings", []))
                suggestions.extend(fabric_results.get("issues", []))
                suggestions.extend(fabric_results.get("suggestions", []))
                suggestions.extend(performance_results.get("optimizations", []))

            else:
                # PySpark best practices checks
                lines = code.split('\n')

                has_spark_imports = any('from pyspark' in line or 'import pyspark' in line for line in lines)
                if not has_spark_imports:
                    warnings.append("No PySpark imports detected. Add: from pyspark.sql import SparkSession")

                has_df_operations = any('df.' in line or '.show()' in line for line in lines)
                if has_df_operations:
                    validation_results.append("DataFrame operations detected")

                if '.collect()' in code:
                    warnings.append(".collect() detected - avoid on large datasets, use .show() or .take() instead")
                if '.toPandas()' in code:
                    warnings.append(".toPandas() detected - ensure dataset fits in driver memory")
                if 'for row in df.collect()' in code:
                    warnings.append("Anti-pattern: iterating over collected DataFrame. Use DataFrame operations instead")

                df_count = code.count('df.')
                if df_count > 3 and '.cache()' not in code and '.persist()' not in code:
                    suggestions.append("Consider caching DataFrame with .cache() for repeated operations")
                if 'createDataFrame' in code and 'StructType' not in code:
                    suggestions.append("Consider defining explicit schema when creating DataFrames")
                if '.filter(' in code and 'isNull' not in code and 'isNotNull' not in code:
                    suggestions.append("Consider adding null value handling in filters")
                if '.write.' in code and 'partitionBy' not in code:
                    suggestions.append("Consider partitioning data when writing large datasets")
                if '.write.' in code and 'format("delta")' not in code:
                    suggestions.append("Consider using Delta Lake format for ACID transactions and time travel")

            # Compile results
            title = "Fabric" if validation_code_type == "fabric" else "PySpark"
            result = f"# {title} Code Validation Report\n\n"
            result += "## Validation Results\n"
            result += "\n".join(f"- {r}" for r in validation_results) + "\n\n"

            if warnings:
                result += "## Warnings\n"
                result += "\n".join(f"- {w}" for w in warnings) + "\n\n"

            if suggestions:
                result += "## Optimization Suggestions\n"
                result += "\n".join(f"- {s}" for s in suggestions) + "\n\n"

            if not warnings and not suggestions:
                result += "## Summary\nCode looks good! No issues detected.\n"
            else:
                result += f"## Summary\nFound {len(warnings)} warnings and {len(suggestions)} optimization opportunities.\n"

            return result

        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            return f"Error validating code: {str(e)}"

    # ── action: analyze_performance ───────────────────────────────────
    elif action == "analyze_performance":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")
            if not notebook_id:
                return "Error: 'notebook_id' is required for 'analyze_performance' action."
            if not workspace:
                return "Error: 'workspace' is required for 'analyze_performance' action."

            # Get notebook content via the get_content action
            notebook_content = await manage_notebook(
                action="get_content",
                workspace=workspace,
                notebook_id=notebook_id,
                ctx=ctx,
            )

            if notebook_content.startswith("Error"):
                return notebook_content

            # Parse notebook and extract code cells
            notebook_data = json.loads(notebook_content)
            cells = notebook_data.get("cells", [])

            code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]

            if not code_cells:
                return "No code cells found in the notebook."

            # Analyze each code cell
            analysis_results = []
            total_operations = 0
            performance_issues = []
            optimization_opportunities = []

            from helpers.pyspark_helpers import PySparkValidator
            validator = PySparkValidator()

            for i, cell in enumerate(code_cells):
                cell_source = "\n".join(cell.get("source", []))

                if not cell_source.strip():
                    continue

                analysis_results.append(f"### Cell {i + 1}")

                # Count operations
                operations = [
                    ("DataFrame reads", cell_source.count("spark.read") + cell_source.count("spark.table")),
                    ("DataFrame writes", cell_source.count(".write.")),
                    ("Transformations", cell_source.count(".withColumn") + cell_source.count(".select") + cell_source.count(".filter")),
                    ("Actions", cell_source.count(".show()") + cell_source.count(".count()") + cell_source.count(".collect()"))
                ]

                for op_name, count in operations:
                    if count > 0:
                        analysis_results.append(f"- {op_name}: {count}")
                        total_operations += count

                # Check for performance patterns
                perf_results = validator.check_performance_patterns(cell_source)
                performance_issues.extend(perf_results["warnings"])
                optimization_opportunities.extend(perf_results["optimizations"])

                # Fabric-specific analysis
                fabric_results = validator.validate_fabric_compatibility(cell_source)
                optimization_opportunities.extend(fabric_results["suggestions"])

            # Generate comprehensive report
            report = f"# Notebook Performance Analysis Report\n\n"
            report += f"**Notebook:** {notebook_id}\n"
            report += f"**Total Code Cells:** {len(code_cells)}\n"
            report += f"**Total Operations:** {total_operations}\n\n"

            if analysis_results:
                report += "## Cell-by-Cell Analysis\n"
                report += "\n".join(analysis_results) + "\n\n"

            if performance_issues:
                report += "## Performance Issues Found\n"
                for issue in set(performance_issues):  # Remove duplicates
                    report += f"- {issue}\n"
                report += "\n"

            if optimization_opportunities:
                report += "## Optimization Opportunities\n"
                for opportunity in set(optimization_opportunities):  # Remove duplicates
                    report += f"- {opportunity}\n"
                report += "\n"

            # Performance score calculation
            score = 100
            score -= len(set(performance_issues)) * 10  # -10 points per unique issue
            score -= len(set(optimization_opportunities)) * 5  # -5 points per optimization opportunity
            score = max(score, 0)  # Ensure score doesn't go negative

            report += f"## Performance Score: {score}/100\n\n"

            if score >= 80:
                report += "Excellent - Your notebook is well-optimized for Fabric!\n"
            elif score >= 60:
                report += "Good - Some optimization opportunities exist.\n"
            elif score >= 40:
                report += "Needs Improvement - Several performance issues should be addressed.\n"
            else:
                report += "Poor - Significant performance optimization required.\n"

            return report

        except Exception as e:
            logger.error(f"Error analyzing notebook performance: {str(e)}")
            return f"Error analyzing notebook performance: {str(e)}"

    else:
        return f"Invalid action '{action}'. Valid actions: list, create, get_content, update_cell, generate_code, validate_code, analyze_performance"


@mcp.tool()
async def manage_spark_jobs(
    action: str,
    workspace: Optional[str] = None,
    notebook: Optional[str] = None,
    job_id: Optional[str] = None,
    scope: Optional[str] = "workspace",
    state: Optional[str] = None,
    item_type: Optional[str] = None,
    limit: Optional[int] = 50,
    ctx: Context = None,
) -> str:
    """Manage Spark job executions (Livy sessions): list jobs or get job details.

    Args:
        action: Action to perform. One of:
            - 'list': List Spark job executions
            - 'get_details': Get detailed information about a specific Spark job
        workspace: Workspace name or ID (uses context if not provided).
            For scope='all', use comma-separated list or omit for all workspaces.
        notebook: Notebook name or ID (optional for 'list' to filter; required for 'get_details')
        job_id: Livy session ID (required for 'get_details')
        scope: 'workspace' (single workspace, default) or 'all' (all accessible workspaces) - 'list' only
        state: Filter by job state: NotStarted, InProgress, Cancelled, Failed, Succeeded - 'list' only
        item_type: Filter by item type: Notebook, SparkJobDefinition, Lakehouse - 'list' only (scope='all')
        limit: Maximum number of jobs to return per workspace (default: 50) - 'list' only
        ctx: Context object containing client information

    Returns:
        Formatted Spark job information or an error message.
    """

    # ── action: list ──────────────────────────────────────────────────
    if action == "list":
        import asyncio

        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")

            fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            spark_client = SparkClient(fabric_client)

            if scope == "all":
                # Multi-workspace mode
                if workspace and workspace.lower() != 'all':
                    workspace_names = [w.strip() for w in workspace.split(',')]
                    workspace_list = []
                    for ws_name in workspace_names:
                        try:
                            ws_id = await fabric_client.resolve_workspace(ws_name)
                            workspace_list.append((ws_id, ws_name))
                        except Exception as e:
                            logger.warning(f"Could not resolve workspace '{ws_name}': {e}")
                else:
                    all_workspaces = await fabric_client.get_workspaces()
                    workspace_list = [(w.get('id'), w.get('displayName')) for w in all_workspaces]

                if not workspace_list:
                    return "No workspaces found or accessible."

                all_jobs = []
                workspace_errors = []

                async def fetch_workspace_jobs(ws_id: str, ws_name: str):
                    try:
                        filters = {}
                        if state:
                            filters["state"] = state
                        sessions = await spark_client.list_workspace_sessions(ws_id, filters, max_results=limit)
                        if sessions:
                            if item_type:
                                sessions = [s for s in sessions if s.get("itemType") == item_type]
                            for s in sessions[:limit]:
                                s['_workspace_id'] = ws_id
                                s['_workspace_name'] = ws_name
                            return sessions[:limit], None
                        return [], None
                    except Exception as e:
                        return [], f"{ws_name}: {str(e)}"

                tasks = [fetch_workspace_jobs(ws_id, ws_name) for ws_id, ws_name in workspace_list]
                results = await asyncio.gather(*tasks)

                for jobs, error in results:
                    if error:
                        workspace_errors.append(error)
                    else:
                        all_jobs.extend(jobs)

                all_jobs.sort(key=lambda x: x.get('submittedDateTime', ''), reverse=True)

                if not all_jobs:
                    filter_info = []
                    if state:
                        filter_info.append(f"state='{state}'")
                    if item_type:
                        filter_info.append(f"type='{item_type}'")
                    filter_msg = f" (filters: {', '.join(filter_info)})" if filter_info else ""
                    return f"No Spark jobs found across {len(workspace_list)} workspaces{filter_msg}."

                markdown = f"# Spark Jobs Across Workspaces\n\n"
                markdown += f"**Workspaces queried:** {len(workspace_list)}\n"
                if state:
                    markdown += f"**State filter:** {state}\n"
                if item_type:
                    markdown += f"**Item type filter:** {item_type}\n"
                markdown += "\n"

                markdown += "| Workspace | Item Name | Type | State | Submitted | Duration | Livy ID |\n"
                markdown += "|-----------|-----------|------|-------|-----------|----------|--------|\n"

                failed_jobs = []
                for session in all_jobs:
                    ws_name = session.get("_workspace_name", "Unknown")
                    item_name = session.get("itemName", "N/A")
                    item_info = session.get("item", {})
                    item_id = item_info.get("id", "") if isinstance(item_info, dict) else ""
                    item_type_val = session.get("itemType", "N/A")
                    job_state = session.get("state", "Unknown")
                    submitted_time = session.get("submittedDateTime", "N/A")
                    livy_id = session.get("livyId", "N/A")

                    if job_state == "Failed":
                        failed_jobs.append({"workspace": ws_name, "item_name": item_name, "item_id": item_id, "livy_id": livy_id})

                    if submitted_time and submitted_time != "N/A":
                        try:
                            dt = datetime.fromisoformat(submitted_time.replace("Z", "+00:00"))
                            submitted_time = dt.strftime("%m-%d %H:%M")
                        except Exception:
                            pass

                    duration = session.get("totalDuration", "N/A")
                    status_emoji = {"Succeeded": "✅", "Failed": "❌", "InProgress": "⏳", "NotStarted": "⏸️", "Cancelled": "🚫"}.get(job_state, "")
                    short_job_id = str(livy_id)[:8] + "..." if len(str(livy_id)) > 12 else livy_id
                    short_ws = ws_name[:15] + "..." if len(ws_name) > 18 else ws_name

                    markdown += f"| {short_ws} | {item_name} | {item_type_val} | {status_emoji} {job_state} | {submitted_time} | {duration} | `{short_job_id}` |\n"

                markdown += f"\n**Total jobs:** {len(all_jobs)}\n"

                if failed_jobs:
                    markdown += f"\n## Failed Jobs ({len(failed_jobs)})\n\n"
                    for fj in failed_jobs[:10]:
                        markdown += f"- **{fj['item_name']}** in `{fj['workspace']}`\n"
                        markdown += f"  - Livy ID: `{fj['livy_id']}`\n"

                # Summary statistics
                state_counts = {}
                for s in all_jobs:
                    st = s.get("state", "Unknown")
                    state_counts[st] = state_counts.get(st, 0) + 1

                if len(state_counts) > 1:
                    markdown += "\n**By state:** "
                    markdown += ", ".join([f"{k}: {v}" for k, v in sorted(state_counts.items())])
                    markdown += "\n"

                if workspace_errors:
                    markdown += f"\n**Errors ({len(workspace_errors)}):**\n"
                    for err in workspace_errors[:5]:
                        markdown += f"- {err}\n"

                return markdown

            else:
                # Single workspace mode
                ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
                if not ws:
                    return "Workspace not set. Use set_context('workspace', ...) or provide workspace parameter."

                workspace_id = await fabric_client.resolve_workspace(ws)

                filters = {}
                if state:
                    filters["state"] = state

                if notebook:
                    if not _is_valid_uuid(notebook):
                        notebook_resolved_id = await fabric_client.resolve_item_id(
                            item=notebook, type="Notebook", workspace=workspace_id
                        )
                    else:
                        notebook_resolved_id = notebook

                    sessions = await spark_client.list_notebook_sessions(workspace_id, notebook_resolved_id)
                    if state and sessions:
                        sessions = [s for s in sessions if s.get("state") == state]
                else:
                    sessions = await spark_client.list_workspace_sessions(workspace_id, filters)

                if not sessions:
                    filter_msg = f" with state '{state}'" if state else ""
                    notebook_msg = f" for notebook '{notebook}'" if notebook else ""
                    return f"No Spark jobs found{notebook_msg}{filter_msg} in workspace '{ws}'."

                markdown = f"# Spark Jobs in workspace '{ws}'\n\n"
                if notebook:
                    markdown = f"# Spark Jobs for notebook '{notebook}' in workspace '{ws}'\n\n"
                if state:
                    markdown += f"**Filtered by state:** {state}\n\n"

                markdown += "| Livy ID | Item Name | Type | State | Submitted | Duration | Operation |\n"
                markdown += "|---------|-----------|------|-------|-----------|----------|----------|\n"

                failed_jobs = []
                for session in sessions:
                    livy_id = session.get("livyId", "N/A")
                    item_name = session.get("itemName", "N/A")
                    item_info = session.get("item", {})
                    item_id = item_info.get("id", "") if isinstance(item_info, dict) else ""
                    item_type_val = session.get("itemType", "N/A")
                    job_state = session.get("state", "Unknown")
                    submitted_time = session.get("submittedDateTime", "N/A")
                    operation_name = session.get("operationName", "N/A")

                    if job_state == "Failed":
                        failed_jobs.append({"item_name": item_name, "item_id": item_id, "livy_id": livy_id})

                    if submitted_time and submitted_time != "N/A":
                        try:
                            dt = datetime.fromisoformat(submitted_time.replace("Z", "+00:00"))
                            submitted_time = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            pass

                    duration = session.get("totalDuration")
                    if not duration:
                        if submitted_time != "N/A":
                            try:
                                submitted_dt = datetime.fromisoformat(session.get("submittedDateTime", "").replace("Z", "+00:00"))
                                end_time = session.get("endDateTime")
                                if end_time:
                                    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                                    duration_seconds = (end_dt - submitted_dt).total_seconds()
                                    duration = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
                                elif job_state in ["InProgress", "NotStarted"]:
                                    now_dt = datetime.now(submitted_dt.tzinfo)
                                    duration_seconds = (now_dt - submitted_dt).total_seconds()
                                    duration = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
                            except Exception:
                                duration = "N/A"
                        else:
                            duration = "N/A"

                    status_emoji = {"Succeeded": "✅", "Failed": "❌", "InProgress": "⏳", "NotStarted": "⏸️", "Cancelled": "🚫"}.get(job_state, "")
                    short_job_id = str(livy_id)[:8] + "..." if len(str(livy_id)) > 12 else livy_id

                    markdown += f"| `{short_job_id}` | {item_name} | {item_type_val} | {status_emoji} {job_state} | {submitted_time} | {duration} | {operation_name} |\n"

                markdown += f"\n**Total jobs:** {len(sessions)}\n"

                state_counts = {}
                for s in sessions:
                    st = s.get("state", "Unknown")
                    state_counts[st] = state_counts.get(st, 0) + 1

                if len(state_counts) > 1:
                    markdown += "\n**By state:** "
                    markdown += ", ".join([f"{k}: {v}" for k, v in sorted(state_counts.items())])
                    markdown += "\n"

                if failed_jobs:
                    markdown += f"\n## Failed Jobs ({len(failed_jobs)})\n\n"
                    markdown += "To get error details, use `manage_spark_jobs` with action='get_details', the Livy ID, and notebook:\n\n"
                    for fj in failed_jobs[:10]:
                        markdown += f"- **{fj['item_name']}**\n"
                        markdown += f"  - Livy ID: `{fj['livy_id']}`\n"

                return markdown

        except Exception as e:
            logger.error(f"Error listing Spark jobs: {str(e)}")
            return f"Error listing Spark jobs: {str(e)}"

    # ── action: get_details ───────────────────────────────────────────
    elif action == "get_details":
        try:
            if ctx is None:
                raise ValueError("Context (ctx) must be provided.")

            if not job_id:
                return "Error: 'job_id' is required for 'get_details' action."

            if not notebook:
                return "Error: 'notebook' parameter is required to get job details."

            fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
            spark_client = SparkClient(fabric_client)

            # Resolve workspace
            ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
            if not ws:
                return "Workspace not set. Please set a workspace using 'set_workspace' command."

            workspace_id = await fabric_client.resolve_workspace(ws)

            # Resolve notebook ID
            if not _is_valid_uuid(notebook):
                notebook_resolved_id = await fabric_client.resolve_item_id(
                    item=notebook, type="Notebook", workspace=workspace_id
                )
            else:
                notebook_resolved_id = notebook

            # Get session details
            session = await spark_client.get_session_details(workspace_id, notebook_resolved_id, job_id)

            if not session:
                return f"No job found with ID '{job_id}' for notebook '{notebook}' in workspace '{ws}'."

            # Log the raw session for debugging
            logger.debug(f"Raw session data: {json.dumps(session, indent=2, default=str)}")

            # Enrich session with resolved names (capacity name, workspace name, etc.)
            try:
                session = await spark_client.enrich_session_with_names(session)
            except Exception as e:
                logger.warning(f"Failed to enrich session with names: {e}")

            # Get notebook metadata to get the display name
            notebook_client = NotebookClient(fabric_client)
            notebook_meta = await notebook_client.get_notebook(workspace_id, notebook_resolved_id)

            # Resolve notebook name - try multiple sources (API field is 'itemName')
            nb_name = session.get('itemName')
            if not nb_name and isinstance(notebook_meta, dict):
                nb_name = notebook_meta.get("displayName")
            if not nb_name:
                nb_name = notebook

            # Get item info from session
            item_info = session.get('item', {})
            item_id = item_info.get('id', notebook_resolved_id)
            job_instance_id = session.get('jobInstanceId')

            # Status information
            job_state = session.get('state', 'Unknown')
            status_emoji = {
                "Succeeded": "✅",
                "Failed": "❌",
                "InProgress": "⏳",
                "NotStarted": "⏸️",
                "Cancelled": "🚫",
                "Unknown": "❓"
            }.get(job_state, "")

            # Format detailed information
            markdown = f"# Spark Job: {nb_name}\n\n"
            markdown += f"**Status:** {status_emoji} {job_state}\n"
            markdown += f"**Operation:** {session.get('operationName', 'N/A')}\n"
            markdown += f"**Workspace:** {ws}\n\n"

            # Timing information - compact format
            submitted = session.get('submittedDateTime', 'N/A')
            started = session.get('startDateTime', 'N/A')
            ended = session.get('endDateTime', 'N/A')
            total_duration = session.get('totalDuration')

            markdown += f"## Timing\n"
            markdown += f"| | |\n|---|---|\n"
            markdown += f"| Submitted | {submitted} |\n"
            markdown += f"| Started | {started} |\n"
            markdown += f"| Ended | {ended} |\n"

            # Show duration
            if total_duration:
                markdown += f"| **Duration** | **{total_duration}** |\n"
            elif submitted != 'N/A' and ended != 'N/A':
                try:
                    submitted_dt = datetime.fromisoformat(submitted.replace("Z", "+00:00"))
                    ended_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                    duration_seconds = (ended_dt - submitted_dt).total_seconds()
                    mins = int(duration_seconds // 60)
                    secs = int(duration_seconds % 60)
                    markdown += f"| **Duration** | **{mins}m {secs}s** |\n"
                except Exception:
                    pass

            # Submitter information - simplified
            submitter = session.get('submitter', {})
            submitter_name = (
                submitter.get('displayName') or
                submitter.get('userPrincipalName') or
                submitter.get('mail') or
                submitter.get('name')
            )
            if submitter_name:
                markdown += f"\n**Submitted by:** {submitter_name}\n"
            elif submitter.get('id'):
                markdown += f"\n**Submitted by:** `{submitter.get('id')}`\n"

            # Cancellation reason if applicable
            if job_state == "Cancelled":
                cancellation_reason = session.get('cancellationReason')
                if cancellation_reason:
                    markdown += f"\n**Cancellation Reason:** {cancellation_reason}\n"

            # Spark Application link
            spark_app_id = session.get('sparkApplicationId')
            if spark_app_id:
                markdown += f"\n**Spark UI:** [Open Spark Application](https://app.fabric.microsoft.com/groups/{workspace_id}/sparkapplication/{spark_app_id})\n"

            # Error information if failed
            job_instance = None
            if job_state == "Failed":
                markdown += f"\n## Error Information\n"

                # Try to get detailed error from job instance API
                if job_instance_id:
                    job_instance = await spark_client.get_job_instance(workspace_id, notebook_resolved_id, job_instance_id)
                    if job_instance:
                        logger.debug(f"Job instance data: {json.dumps(job_instance, indent=2, default=str)}")
                        failure_reason = (
                            job_instance.get('failureReason') or
                            job_instance.get('errorMessage') or
                            job_instance.get('error') or
                            job_instance.get('message')
                        )
                        if failure_reason:
                            markdown += f"**Failure Reason:**\n```\n{failure_reason}\n```\n\n"

                        root_activity_id = job_instance.get('rootActivityId')
                        if root_activity_id:
                            markdown += f"**Root Activity ID:** `{root_activity_id}`\n"

                # Include any errors from the session response
                errors = session.get('errors') or session.get('error') or []
                if isinstance(errors, dict):
                    errors = [errors]
                if errors:
                    markdown += f"\n**Error Details:**\n"
                    for error in errors:
                        if isinstance(error, str):
                            markdown += f"```\n{error}\n```\n"
                        else:
                            error_code = error.get('errorCode') or error.get('code') or 'Unknown'
                            error_message = error.get('message') or error.get('errorMessage') or 'No message'
                            error_source = error.get('source') or ''
                            markdown += f"- **{error_code}**"
                            if error_source:
                                markdown += f" (Source: {error_source})"
                            markdown += f"\n  ```\n  {error_message}\n  ```\n"

                # If no specific errors found
                has_failure_info = bool(errors) or (job_instance and (job_instance.get('failureReason') or job_instance.get('errorMessage')))
                if not has_failure_info:
                    markdown += "\n*No detailed error information available from the API.*\n"
                    markdown += "*Check the Spark logs in the Fabric portal for more details.*\n"

            # Links section
            markdown += f"\n## Links\n"
            markdown += f"- [Open Notebook](https://app.fabric.microsoft.com/groups/{workspace_id}/notebooks/{notebook_resolved_id})\n"
            if job_instance_id:
                markdown += f"- [View in Monitoring Hub](https://app.fabric.microsoft.com/groups/{workspace_id}/monitoringhub?experience=fabric-developer&jobId={job_instance_id})\n"

            return markdown

        except Exception as e:
            logger.error(f"Error getting job details: {str(e)}")
            return f"Error getting job details: {str(e)}"

    else:
        return f"Invalid action '{action}'. Valid actions: list, get_details"


# ===== HELPER FUNCTIONS =====

def _get_pyspark_template(template_type: str) -> Optional[dict]:
    """Get a PySpark notebook template by type.

    Args:
        template_type: Type of PySpark template ('basic', 'etl', 'analytics', 'ml')

    Returns:
        Notebook JSON dict or None if template_type is invalid.
    """
    templates = {
            "basic": {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# PySpark Notebook\n",
                            "\n",
                            "This notebook demonstrates basic PySpark operations in Microsoft Fabric.\n"
                        ],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Initialize Spark session\n",
                            "from pyspark.sql import SparkSession\n",
                            "from pyspark.sql.functions import *\n",
                            "from pyspark.sql.types import *\n",
                            "\n",
                            "# Spark session is already available as 'spark' in Fabric\n",
                            "print(f\"Spark version: {spark.version}\")\n",
                            "print(f\"Available cores: {spark.sparkContext.defaultParallelism}\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Sample data creation\n",
                            "sample_data = [\n",
                            "    (1, \"John\", 25, \"Engineering\"),\n",
                            "    (2, \"Jane\", 30, \"Marketing\"),\n",
                            "    (3, \"Bob\", 35, \"Sales\"),\n",
                            "    (4, \"Alice\", 28, \"Engineering\")\n",
                            "]\n",
                            "\n",
                            "schema = StructType([\n",
                            "    StructField(\"id\", IntegerType(), True),\n",
                            "    StructField(\"name\", StringType(), True),\n",
                            "    StructField(\"age\", IntegerType(), True),\n",
                            "    StructField(\"department\", StringType(), True)\n",
                            "])\n",
                            "\n",
                            "df = spark.createDataFrame(sample_data, schema)\n",
                            "df.show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    }
                ]
            },
            "etl": {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# PySpark ETL Pipeline\n",
                            "\n",
                            "This notebook demonstrates an ETL pipeline using PySpark in Microsoft Fabric.\n"
                        ],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Import necessary libraries\n",
                            "from pyspark.sql import SparkSession\n",
                            "from pyspark.sql.functions import *\n",
                            "from pyspark.sql.types import *\n",
                            "from delta.tables import DeltaTable\n",
                            "\n",
                            "print(f\"Spark version: {spark.version}\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Extract: Read data from source\n",
                            "# Example: Reading from a lakehouse table\n",
                            "# df_source = spark.table(\"lakehouse.table_name\")\n",
                            "\n",
                            "# For demo purposes, create sample data\n",
                            "raw_data = [\n",
                            "    (\"2024-01-01\", \"Product A\", 100, 25.50),\n",
                            "    (\"2024-01-01\", \"Product B\", 150, 30.00),\n",
                            "    (\"2024-01-02\", \"Product A\", 120, 25.50),\n",
                            "    (\"2024-01-02\", \"Product C\", 80, 45.00)\n",
                            "]\n",
                            "\n",
                            "schema = StructType([\n",
                            "    StructField(\"date\", StringType(), True),\n",
                            "    StructField(\"product\", StringType(), True),\n",
                            "    StructField(\"quantity\", IntegerType(), True),\n",
                            "    StructField(\"price\", DoubleType(), True)\n",
                            "])\n",
                            "\n",
                            "df_raw = spark.createDataFrame(raw_data, schema)\n",
                            "print(\"Raw data:\")\n",
                            "df_raw.show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Transform: Clean and process data\n",
                            "df_transformed = df_raw \\\n",
                            "    .withColumn(\"date\", to_date(col(\"date\"), \"yyyy-MM-dd\")) \\\n",
                            "    .withColumn(\"revenue\", col(\"quantity\") * col(\"price\")) \\\n",
                            "    .withColumn(\"year\", year(col(\"date\"))) \\\n",
                            "    .withColumn(\"month\", month(col(\"date\")))\n",
                            "\n",
                            "print(\"Transformed data:\")\n",
                            "df_transformed.show()\n",
                            "df_transformed.printSchema()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Load: Write processed data to target\n",
                            "# Example: Writing to a Delta table in lakehouse\n",
                            "# df_transformed.write \\\n",
                            "#     .format(\"delta\") \\\n",
                            "#     .mode(\"overwrite\") \\\n",
                            "#     .saveAsTable(\"lakehouse.processed_sales\")\n",
                            "\n",
                            "print(\"ETL pipeline completed successfully!\")\n",
                            "print(f\"Processed {df_transformed.count()} records\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    }
                ]
            },
            "analytics": {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# PySpark Data Analytics\n",
                            "\n",
                            "This notebook demonstrates data analytics using PySpark in Microsoft Fabric.\n"
                        ],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Import libraries for analytics\n",
                            "from pyspark.sql import SparkSession\n",
                            "from pyspark.sql.functions import *\n",
                            "from pyspark.sql.types import *\n",
                            "from pyspark.sql.window import Window\n",
                            "\n",
                            "print(f\"Spark version: {spark.version}\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Create sample sales data for analytics\n",
                            "sales_data = [\n",
                            "    (\"2024-01-01\", \"North\", \"Product A\", 1000, 100),\n",
                            "    (\"2024-01-01\", \"South\", \"Product A\", 800, 80),\n",
                            "    (\"2024-01-02\", \"North\", \"Product B\", 1200, 120),\n",
                            "    (\"2024-01-02\", \"South\", \"Product B\", 900, 90),\n",
                            "    (\"2024-01-03\", \"East\", \"Product A\", 1100, 110),\n",
                            "    (\"2024-01-03\", \"West\", \"Product C\", 700, 70)\n",
                            "]\n",
                            "\n",
                            "schema = StructType([\n",
                            "    StructField(\"date\", StringType(), True),\n",
                            "    StructField(\"region\", StringType(), True),\n",
                            "    StructField(\"product\", StringType(), True),\n",
                            "    StructField(\"revenue\", IntegerType(), True),\n",
                            "    StructField(\"quantity\", IntegerType(), True)\n",
                            "])\n",
                            "\n",
                            "df_sales = spark.createDataFrame(sales_data, schema)\n",
                            "df_sales = df_sales.withColumn(\"date\", to_date(col(\"date\"), \"yyyy-MM-dd\"))\n",
                            "df_sales.show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Aggregation analysis\n",
                            "print(\"=== Revenue by Region ===\")\n",
                            "df_sales.groupBy(\"region\") \\\n",
                            "    .agg(sum(\"revenue\").alias(\"total_revenue\"),\n",
                            "         sum(\"quantity\").alias(\"total_quantity\"),\n",
                            "         count(\"*\").alias(\"transaction_count\")) \\\n",
                            "    .orderBy(desc(\"total_revenue\")) \\\n",
                            "    .show()\n",
                            "\n",
                            "print(\"=== Revenue by Product ===\")\n",
                            "df_sales.groupBy(\"product\") \\\n",
                            "    .agg(sum(\"revenue\").alias(\"total_revenue\"),\n",
                            "         avg(\"revenue\").alias(\"avg_revenue\")) \\\n",
                            "    .orderBy(desc(\"total_revenue\")) \\\n",
                            "    .show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Window functions for advanced analytics\n",
                            "windowSpec = Window.partitionBy(\"region\").orderBy(\"date\")\n",
                            "\n",
                            "df_analytics = df_sales \\\n",
                            "    .withColumn(\"running_total\", sum(\"revenue\").over(windowSpec)) \\\n",
                            "    .withColumn(\"row_number\", row_number().over(windowSpec)) \\\n",
                            "    .withColumn(\"rank\", rank().over(windowSpec.orderBy(desc(\"revenue\"))))\n",
                            "\n",
                            "print(\"=== Advanced Analytics with Window Functions ===\")\n",
                            "df_analytics.select(\"date\", \"region\", \"product\", \"revenue\", \n",
                            "                   \"running_total\", \"row_number\", \"rank\") \\\n",
                            "    .orderBy(\"region\", \"date\") \\\n",
                            "    .show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    }
                ]
            },
            "ml": {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# PySpark Machine Learning\n",
                            "\n",
                            "This notebook demonstrates machine learning with PySpark MLlib in Microsoft Fabric.\n"
                        ],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Import ML libraries\n",
                            "from pyspark.sql import SparkSession\n",
                            "from pyspark.sql.functions import *\n",
                            "from pyspark.sql.types import *\n",
                            "from pyspark.ml.feature import VectorAssembler, StandardScaler\n",
                            "from pyspark.ml.regression import LinearRegression\n",
                            "from pyspark.ml.evaluation import RegressionEvaluator\n",
                            "from pyspark.ml import Pipeline\n",
                            "\n",
                            "print(f\"Spark version: {spark.version}\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Create sample dataset for regression\n",
                            "ml_data = [\n",
                            "    (1, 2.0, 3.0, 4.0, 10.0),\n",
                            "    (2, 3.0, 4.0, 5.0, 15.0),\n",
                            "    (3, 4.0, 5.0, 6.0, 20.0),\n",
                            "    (4, 5.0, 6.0, 7.0, 25.0),\n",
                            "    (5, 6.0, 7.0, 8.0, 30.0),\n",
                            "    (6, 7.0, 8.0, 9.0, 35.0)\n",
                            "]\n",
                            "\n",
                            "schema = StructType([\n",
                            "    StructField(\"id\", IntegerType(), True),\n",
                            "    StructField(\"feature1\", DoubleType(), True),\n",
                            "    StructField(\"feature2\", DoubleType(), True),\n",
                            "    StructField(\"feature3\", DoubleType(), True),\n",
                            "    StructField(\"label\", DoubleType(), True)\n",
                            "])\n",
                            "\n",
                            "df_ml = spark.createDataFrame(ml_data, schema)\n",
                            "print(\"Sample ML dataset:\")\n",
                            "df_ml.show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Feature engineering pipeline\n",
                            "feature_cols = [\"feature1\", \"feature2\", \"feature3\"]\n",
                            "\n",
                            "# Assemble features into a vector\n",
                            "assembler = VectorAssembler(inputCols=feature_cols, outputCol=\"raw_features\")\n",
                            "\n",
                            "# Scale features\n",
                            "scaler = StandardScaler(inputCol=\"raw_features\", outputCol=\"features\")\n",
                            "\n",
                            "# Linear regression model\n",
                            "lr = LinearRegression(featuresCol=\"features\", labelCol=\"label\")\n",
                            "\n",
                            "# Create pipeline\n",
                            "pipeline = Pipeline(stages=[assembler, scaler, lr])\n",
                            "\n",
                            "print(\"ML Pipeline created with stages: Feature Assembly -> Scaling -> Linear Regression\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Split data and train model\n",
                            "train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)\n",
                            "\n",
                            "print(f\"Training data count: {train_data.count()}\")\n",
                            "print(f\"Test data count: {test_data.count()}\")\n",
                            "\n",
                            "# Train the pipeline\n",
                            "model = pipeline.fit(train_data)\n",
                            "\n",
                            "# Make predictions\n",
                            "predictions = model.transform(test_data)\n",
                            "\n",
                            "print(\"\\nPredictions:\")\n",
                            "predictions.select(\"id\", \"label\", \"prediction\").show()\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "# Evaluate model performance\n",
                            "evaluator = RegressionEvaluator(labelCol=\"label\", predictionCol=\"prediction\", metricName=\"rmse\")\n",
                            "rmse = evaluator.evaluate(predictions)\n",
                            "\n",
                            "evaluator_r2 = RegressionEvaluator(labelCol=\"label\", predictionCol=\"prediction\", metricName=\"r2\")\n",
                            "r2 = evaluator_r2.evaluate(predictions)\n",
                            "\n",
                            "print(f\"Root Mean Square Error (RMSE): {rmse:.3f}\")\n",
                            "print(f\"R-squared (R2): {r2:.3f}\")\n",
                            "\n",
                            "# Get model coefficients\n",
                            "lr_model = model.stages[-1]\n",
                            "print(f\"\\nModel coefficients: {lr_model.coefficients}\")\n",
                            "print(f\"Model intercept: {lr_model.intercept:.3f}\")\n"
                        ],
                        "execution_count": None,
                        "outputs": [],
                        "metadata": {}
                    }
                ]
            }
        }

    if template_type not in templates:
        return None

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": templates[template_type]["cells"],
        "metadata": {
            "language_info": {"name": "python"},
            "kernel_info": {"name": "synapse_pyspark"},
            "description": f"PySpark notebook created from {template_type} template"
        },
    }


def _parse_fabric_py_to_cells(py_content: str) -> List[Dict[str, Any]]:
    """Parse Fabric's native notebook-content.py format into cells.

    Fabric notebooks in .py format use special markers:
    - # METADATA **{...}** - notebook metadata
    - # MARKDOWN **...**  - markdown cells
    - # CELL **{...}**    - code cell metadata
    - # PARAMETERS        - parameter cell marker

    Args:
        py_content: The Python content from notebook-content.py

    Returns:
        List of cell dictionaries in ipynb format
    """
    cells = []
    lines = py_content.split('\n')
    current_cell = None
    current_source = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for markdown cell
        if line.strip().startswith('# MARKDOWN **'):
            # Save previous cell
            if current_cell is not None:
                current_cell["source"] = current_source
                cells.append(current_cell)

            # Extract markdown content (may span multiple lines)
            markdown_content = []
            # Get content after MARKDOWN **
            content_start = line.find('# MARKDOWN **') + len('# MARKDOWN **')
            first_line = line[content_start:]

            if first_line.endswith('**'):
                # Single line markdown
                markdown_content.append(first_line[:-2])
            else:
                markdown_content.append(first_line)
                i += 1
                while i < len(lines) and not lines[i].rstrip().endswith('**'):
                    markdown_content.append(lines[i])
                    i += 1
                if i < len(lines):
                    # Last line, remove trailing **
                    last_line = lines[i].rstrip()
                    if last_line.endswith('**'):
                        markdown_content.append(last_line[:-2])

            current_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + '\n' for l in markdown_content[:-1]] + [markdown_content[-1]] if markdown_content else []
            }
            cells.append(current_cell)
            current_cell = None
            current_source = []

        # Check for code cell marker
        elif line.strip().startswith('# CELL **'):
            # Save previous cell
            if current_cell is not None:
                current_cell["source"] = current_source
                cells.append(current_cell)

            # Parse cell metadata if present
            metadata = {}
            try:
                meta_start = line.find('# CELL **') + len('# CELL **')
                meta_end = line.rfind('**')
                if meta_end > meta_start:
                    meta_str = line[meta_start:meta_end]
                    metadata = json.loads(meta_str)
            except (json.JSONDecodeError, ValueError):
                pass

            current_cell = {
                "cell_type": "code",
                "execution_count": None,
                "outputs": [],
                "metadata": metadata,
            }
            current_source = []

        # Check for metadata line (skip)
        elif line.strip().startswith('# METADATA **'):
            pass

        # Check for parameters marker
        elif line.strip() == '# PARAMETERS':
            if current_cell is None:
                current_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "outputs": [],
                    "metadata": {"tags": ["parameters"]},
                }
                current_source = []

        # Regular code line
        elif current_cell is not None:
            current_source.append(line + '\n')

        # Start of first cell if no marker seen yet
        elif line.strip() and not line.strip().startswith('#'):
            current_cell = {
                "cell_type": "code",
                "execution_count": None,
                "outputs": [],
                "metadata": {},
            }
            current_source = [line + '\n']

        i += 1

    # Don't forget the last cell
    if current_cell is not None:
        # Remove trailing empty lines from source
        while current_source and current_source[-1].strip() == '':
            current_source.pop()
        if current_source:
            # Remove newline from last line
            current_source[-1] = current_source[-1].rstrip('\n')
        current_cell["source"] = current_source
        cells.append(current_cell)

    return cells


def _cells_to_fabric_py(cells: List[Dict[str, Any]]) -> str:
    """Convert cells back to Fabric's native notebook-content.py format.

    Args:
        cells: List of cell dictionaries in ipynb format

    Returns:
        Python content string in Fabric format
    """
    output_lines = []

    for cell in cells:
        cell_type = cell.get("cell_type", "code")
        source = cell.get("source", [])

        # Convert source to string if it's a list
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source

        if cell_type == "markdown":
            # Wrap in MARKDOWN markers
            output_lines.append(f"# MARKDOWN **{source_str}**")
            output_lines.append("")

        elif cell_type == "code":
            metadata = cell.get("metadata", {})

            # Add CELL marker with metadata
            if metadata:
                output_lines.append(f"# CELL **{json.dumps(metadata)}**")
            else:
                output_lines.append("# CELL **{}**")

            # Check for parameters tag
            if "parameters" in metadata.get("tags", []):
                output_lines.append("# PARAMETERS")

            output_lines.append(source_str.rstrip())
            output_lines.append("")

    return '\n'.join(output_lines)
