"""Lakehouse orchestration for data exploration workflows."""

from helpers.logging_config import get_logger
from typing import Dict, Any, Optional, List
import json
import time

logger = get_logger(__name__)


class LakehouseOrchestrator:
    """Orchestrates complex lakehouse data exploration workflows."""

    def __init__(self):
        self._relationship_engine = None

    @property
    def relationship_engine(self):
        """Lazy load relationship engine to avoid circular imports."""
        if self._relationship_engine is None:
            from .tool_relationships import relationship_engine
            self._relationship_engine = relationship_engine
        return self._relationship_engine

    async def explore_lakehouse_complete(
        self,
        workspace: str,
        lakehouse: Optional[str] = None,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Comprehensive lakehouse exploration workflow.

        Args:
            workspace: Workspace name
            lakehouse: Specific lakehouse name (optional, will explore all if not provided)
            ctx: Context object

        Returns:
            Complete exploration results with suggestions
        """
        from tools.lakehouse import list_lakehouses
        from tools.table import list_tables, get_all_lakehouse_schemas

        results = {
            'workflow': 'explore_lakehouse_complete',
            'steps_completed': [],
            'timestamp': time.time(),
            'workspace': workspace
        }

        try:
            # Step 1: List lakehouses if specific one not provided
            if not lakehouse:
                logger.info(f"[Orchestrator] Listing lakehouses in workspace: {workspace}")
                lakehouses_result = await list_lakehouses(workspace=workspace, ctx=ctx)
                lakehouses_data = json.loads(lakehouses_result) if isinstance(lakehouses_result, str) else lakehouses_result

                results['lakehouses'] = lakehouses_data
                results['steps_completed'].append('list_lakehouses')

                # Select first lakehouse if available
                lakehouses_list = lakehouses_data.get('lakehouses', [])
                if lakehouses_list:
                    lakehouse = lakehouses_list[0] if isinstance(lakehouses_list[0], str) else lakehouses_list[0].get('name')
                    results['selected_lakehouse'] = lakehouse
                else:
                    results['error'] = 'No lakehouses found in workspace'
                    results['success'] = False
                    results['suggested_next_actions'] = [
                        {
                            'tool': 'create_lakehouse',
                            'priority': 10,
                            'reason': 'No lakehouses found - create one first',
                            'params': {'workspace': workspace}
                        }
                    ]
                    return results

            results['target_lakehouse'] = lakehouse

            # Step 2: List tables
            logger.info(f"[Orchestrator] Listing tables in lakehouse: {lakehouse}")
            tables_result = await list_tables(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )
            tables_data = json.loads(tables_result) if isinstance(tables_result, str) else tables_result

            results['tables'] = tables_data
            results['steps_completed'].append('list_tables')

            tables_list = tables_data.get('tables', [])
            if not tables_list:
                results['warning'] = 'No tables found in lakehouse'
                results['suggested_next_actions'] = [
                    {
                        'tool': 'create_pyspark_notebook',
                        'priority': 8,
                        'reason': 'Create ETL notebook to load data into lakehouse',
                        'params': {'workspace': workspace, 'template_type': 'etl'}
                    }
                ]
                results['success'] = True
                return results

            results['table_count'] = len(tables_list)

            # Step 3: Get schemas for all tables
            logger.info(f"[Orchestrator] Getting schemas for {len(tables_list)} tables")
            schemas_result = await get_all_lakehouse_schemas(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )
            schemas_data = json.loads(schemas_result) if isinstance(schemas_result, str) else schemas_result

            results['schemas'] = schemas_data
            results['steps_completed'].append('get_schemas')

            # Step 4: Generate sample query code
            sample_table = tables_list[0] if tables_list else None
            if sample_table:
                table_name = sample_table if isinstance(sample_table, str) else sample_table.get('name')
                results['sample_code'] = self._generate_sample_code(table_name, lakehouse)
                results['steps_completed'].append('generate_sample')

            # Step 5: Generate intelligent suggestions
            results['suggested_next_actions'] = self._suggest_next_steps(
                workspace=workspace,
                lakehouse=lakehouse,
                table_count=len(tables_list),
                schemas=schemas_data
            )

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Lakehouse exploration error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    async def setup_data_integration(
        self,
        workspace: str,
        lakehouse: str,
        goal: str,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Setup data integration workflow.

        Args:
            workspace: Workspace name
            lakehouse: Lakehouse name
            goal: Integration goal description
            ctx: Context object

        Returns:
            Integration setup results
        """
        from tools.table import list_tables, get_lakehouse_table_schema
        from tools.notebook import create_pyspark_notebook, generate_pyspark_code

        results = {
            'workflow': 'setup_data_integration',
            'steps_completed': [],
            'workspace': workspace,
            'lakehouse': lakehouse,
            'goal': goal
        }

        try:
            # Analyze goal to determine integration type
            goal_lower = goal.lower()
            is_read = any(keyword in goal_lower for keyword in ['read', 'load', 'import', 'fetch'])
            is_write = any(keyword in goal_lower for keyword in ['write', 'save', 'export', 'store'])

            # List tables
            tables_result = await list_tables(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )
            tables_data = json.loads(tables_result) if isinstance(tables_result, str) else tables_result
            results['tables'] = tables_data
            results['steps_completed'].append('list_tables')

            tables_list = tables_data.get('tables', [])
            if not tables_list and is_read:
                results['error'] = 'No tables found to read from'
                results['suggested_next_actions'] = [
                    {
                        'tool': 'create_pyspark_notebook',
                        'priority': 9,
                        'reason': 'Create notebook to load data into lakehouse first',
                        'params': {'workspace': workspace, 'template_type': 'etl'}
                    }
                ]
                results['success'] = False
                return results

            # Create integration notebook
            notebook_name = f"integration_{lakehouse}_{int(time.time())}"
            logger.info(f"[Orchestrator] Creating integration notebook: {notebook_name}")

            notebook_result = await create_pyspark_notebook(
                workspace=workspace,
                notebook_name=notebook_name,
                template_type='etl' if is_read else 'fabric_integration',
                ctx=ctx
            )
            notebook_data = json.loads(notebook_result) if isinstance(notebook_result, str) else notebook_result
            results['notebook'] = notebook_data
            results['steps_completed'].append('create_notebook')

            # Generate integration code
            if is_read and tables_list:
                table_name = tables_list[0] if isinstance(tables_list[0], str) else tables_list[0].get('name')
                code_result = await generate_pyspark_code(
                    operation='read_table',
                    source_table=table_name,
                    lakehouse=lakehouse,
                    ctx=ctx
                )
                results['generated_code'] = json.loads(code_result) if isinstance(code_result, str) else code_result
                results['steps_completed'].append('generate_read_code')

            elif is_write:
                code_result = await generate_pyspark_code(
                    operation='write_table',
                    target_table='your_target_table',
                    lakehouse=lakehouse,
                    ctx=ctx
                )
                results['generated_code'] = json.loads(code_result) if isinstance(code_result, str) else code_result
                results['steps_completed'].append('generate_write_code')

            results['suggested_next_actions'] = [
                {
                    'tool': 'validate_pyspark_code',
                    'priority': 8,
                    'reason': 'Validate integration code before execution',
                    'params': {}
                },
                {
                    'tool': 'generate_pyspark_code',
                    'priority': 7,
                    'reason': 'Add data quality checks',
                    'params': {'operation': 'data_quality'}
                }
            ]

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Data integration setup error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    async def analyze_lakehouse_performance(
        self,
        workspace: str,
        lakehouse: str,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Analyze lakehouse performance and structure.

        Args:
            workspace: Workspace name
            lakehouse: Lakehouse name
            ctx: Context object

        Returns:
            Performance analysis results
        """
        from tools.table import list_tables, get_all_lakehouse_schemas
        from tools.lakehouse import run_lakehouse_query

        results = {
            'workflow': 'analyze_lakehouse_performance',
            'steps_completed': [],
            'workspace': workspace,
            'lakehouse': lakehouse
        }

        try:
            # Get tables
            tables_result = await list_tables(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )
            tables_data = json.loads(tables_result) if isinstance(tables_result, str) else tables_result

            tables_list = tables_data.get('tables', [])
            results['table_count'] = len(tables_list)
            results['steps_completed'].append('count_tables')

            # Get schemas
            schemas_result = await get_all_lakehouse_schemas(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )
            schemas_data = json.loads(schemas_result) if isinstance(schemas_result, str) else schemas_result

            results['schemas'] = schemas_data
            results['steps_completed'].append('get_schemas')

            # Analyze structure
            total_columns = 0
            complex_types = 0
            for table_schema in schemas_data.get('tables', []):
                columns = table_schema.get('columns', [])
                total_columns += len(columns)
                for col in columns:
                    col_type = col.get('type', '').lower()
                    if any(t in col_type for t in ['array', 'struct', 'map']):
                        complex_types += 1

            results['analysis'] = {
                'total_tables': len(tables_list),
                'total_columns': total_columns,
                'avg_columns_per_table': total_columns / len(tables_list) if tables_list else 0,
                'complex_types_count': complex_types,
            }
            results['steps_completed'].append('analyze_structure')

            # Generate suggestions
            suggestions = []
            if len(tables_list) > 50:
                suggestions.append({
                    'tool': 'generate_pyspark_code',
                    'priority': 7,
                    'reason': f'Large lakehouse ({len(tables_list)} tables) - consider partitioning strategies',
                    'params': {'operation': 'partitioning'}
                })

            if complex_types > 10:
                suggestions.append({
                    'tool': 'generate_pyspark_code',
                    'priority': 6,
                    'reason': f'{complex_types} complex types detected - review schema optimization',
                    'params': {'operation': 'schema_optimization'}
                })

            results['suggested_next_actions'] = suggestions
            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Lakehouse performance analysis error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    def _generate_sample_code(self, table_name: str, lakehouse: str) -> str:
        """Generate sample PySpark code for reading a table."""
        return f"""# Sample code to read from {table_name}
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("Read{table_name}").getOrCreate()

# Read from Delta table
df = spark.read.format("delta").table("{lakehouse}.{table_name}")

# Display sample data
df.show(10)

# Get schema
df.printSchema()

# Basic statistics
df.describe().show()
"""

    def _suggest_next_steps(
        self,
        workspace: str,
        lakehouse: str,
        table_count: int,
        schemas: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate context-aware suggestions for lakehouse exploration."""
        suggestions = []

        if table_count == 0:
            suggestions.append({
                'tool': 'create_pyspark_notebook',
                'priority': 9,
                'reason': 'No tables found - create ETL notebook to load data',
                'params': {'workspace': workspace, 'template_type': 'etl'}
            })
        elif table_count < 5:
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'priority': 8,
                'reason': 'Generate code to read from discovered tables',
                'params': {'operation': 'read_table', 'lakehouse': lakehouse}
            })
            suggestions.append({
                'tool': 'create_pyspark_notebook',
                'priority': 7,
                'reason': 'Create analytics notebook to analyze data',
                'params': {'workspace': workspace, 'template_type': 'analytics'}
            })
        else:
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'priority': 8,
                'reason': f'{table_count} tables found - generate ETL code',
                'params': {'operation': 'etl', 'lakehouse': lakehouse}
            })
            suggestions.append({
                'tool': 'create_pyspark_notebook',
                'priority': 7,
                'reason': 'Create comprehensive data pipeline notebook',
                'params': {'workspace': workspace, 'template_type': 'etl'}
            })

        # Always suggest data quality checks
        suggestions.append({
            'tool': 'generate_pyspark_code',
            'priority': 6,
            'reason': 'Add data quality validation',
            'params': {'operation': 'data_quality', 'lakehouse': lakehouse}
        })

        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)[:5]


# Singleton instance
lakehouse_orchestrator = LakehouseOrchestrator()
