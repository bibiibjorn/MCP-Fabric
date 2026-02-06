"""Notebook orchestration for multi-step workflows."""

from helpers.utils.context import get_context
from helpers.logging_config import get_logger
from typing import Dict, Any, Optional, List
import json
import time

logger = get_logger(__name__)


class NotebookOrchestrator:
    """Orchestrates complex notebook development workflows."""

    def __init__(self):
        self._relationship_engine = None

    @property
    def relationship_engine(self):
        """Lazy load relationship engine to avoid circular imports."""
        if self._relationship_engine is None:
            from .tool_relationships import relationship_engine
            self._relationship_engine = relationship_engine
        return self._relationship_engine

    async def create_validated_notebook(
        self,
        workspace: str,
        notebook_name: str,
        template_type: str,
        validate: bool = True,
        optimize: bool = False,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Create notebook with automatic validation and optimization.

        Args:
            workspace: Fabric workspace name
            notebook_name: Name for new notebook
            template_type: Template (basic/etl/analytics/ml/fabric_integration/streaming)
            validate: Run validation after creation
            optimize: Run optimization analysis
            ctx: Context object

        Returns:
            Dict with creation results, validation, and suggestions
        """
        from tools.notebook import (
            create_pyspark_notebook,
            validate_pyspark_code,
            analyze_notebook_performance
        )

        results = {
            'workflow': 'create_validated_notebook',
            'steps_completed': [],
            'timestamp': time.time()
        }

        try:
            # Step 1: Create notebook
            logger.info(f"[Orchestrator] Creating notebook: {notebook_name}")
            creation_result = await create_pyspark_notebook(
                workspace=workspace,
                notebook_name=notebook_name,
                template_type=template_type,
                ctx=ctx
            )

            # Parse result
            try:
                creation_data = json.loads(creation_result) if isinstance(creation_result, str) else creation_result
            except (json.JSONDecodeError, TypeError, ValueError):
                creation_data = {'raw_response': creation_result}

            results['notebook_created'] = creation_data
            results['steps_completed'].append('create')

            # Extract notebook ID
            notebook_id = creation_data.get('notebook_id') or creation_data.get('id')

            if not notebook_id:
                results['error'] = 'Failed to get notebook_id from creation'
                results['success'] = False
                return results

            results['notebook_id'] = notebook_id

            # Step 2: Validate (conditional)
            if validate:
                logger.info(f"[Orchestrator] Validating notebook: {notebook_id}")

                # Get notebook content first
                from tools.notebook import get_notebook_content
                try:
                    content_result = await get_notebook_content(
                        workspace=workspace,
                        notebook_id=notebook_id,
                        ctx=ctx
                    )
                    content_data = json.loads(content_result) if isinstance(content_result, str) else content_result

                    # Extract code cells
                    code = self._extract_code_from_notebook(content_data)

                    # Validate
                    validation_result = await validate_pyspark_code(code=code, ctx=ctx)
                    validation_data = json.loads(validation_result) if isinstance(validation_result, str) else validation_result

                    results['validation'] = validation_data
                    results['steps_completed'].append('validate')

                    # Check for critical errors
                    if validation_data.get('has_errors', False):
                        results['warning'] = 'Validation found errors'
                        results['suggested_next_actions'] = [
                            {
                                'tool': 'generate_pyspark_code',
                                'reason': 'Fix validation errors',
                                'priority': 10,
                                'params': {'operation': 'fix_errors'}
                            }
                        ]
                        results['success'] = True  # Still successful, just with warnings
                        return results
                except Exception as e:
                    logger.warning(f"[Orchestrator] Validation step failed: {e}")
                    results['validation_error'] = str(e)

            # Step 3: Optimize (conditional)
            if optimize:
                logger.info(f"[Orchestrator] Analyzing performance: {notebook_id}")
                try:
                    performance_result = await analyze_notebook_performance(
                        workspace=workspace,
                        notebook_id=notebook_id,
                        ctx=ctx
                    )
                    performance_data = json.loads(performance_result) if isinstance(performance_result, str) else performance_result

                    results['performance'] = performance_data
                    results['steps_completed'].append('optimize')
                except Exception as e:
                    logger.warning(f"[Orchestrator] Performance analysis failed: {e}")
                    results['performance_error'] = str(e)

            # Step 4: Add intelligent suggestions
            results['suggested_next_actions'] = self._suggest_next_steps(
                template_type=template_type,
                validation=results.get('validation'),
                performance=results.get('performance'),
                notebook_id=notebook_id,
                workspace=workspace
            )

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Orchestration error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    async def validate_and_suggest(
        self,
        notebook_id: str,
        workspace: str,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Validate notebook and provide intelligent suggestions.

        Args:
            notebook_id: Notebook ID to validate
            workspace: Workspace name
            ctx: Context object

        Returns:
            Validation results with suggestions
        """
        from tools.notebook import get_notebook_content, validate_pyspark_code

        results = {
            'workflow': 'validate_and_suggest',
            'steps_completed': [],
            'notebook_id': notebook_id
        }

        try:
            # Get content
            content_result = await get_notebook_content(
                workspace=workspace,
                notebook_id=notebook_id,
                ctx=ctx
            )
            content_data = json.loads(content_result) if isinstance(content_result, str) else content_result
            code = self._extract_code_from_notebook(content_data)

            # Validate
            validation_result = await validate_pyspark_code(code=code, ctx=ctx)
            validation_data = json.loads(validation_result) if isinstance(validation_result, str) else validation_result

            results['validation'] = validation_data
            results['steps_completed'].append('validate')

            # Generate suggestions based on validation
            suggestions = []
            if validation_data.get('has_errors'):
                suggestions.append({
                    'tool': 'generate_pyspark_code',
                    'priority': 10,
                    'reason': 'Fix validation errors',
                    'params': {'operation': 'fix_errors'}
                })
            else:
                suggestions.append({
                    'tool': 'analyze_notebook_performance',
                    'priority': 8,
                    'reason': 'Validation passed - analyze performance',
                    'params': {'notebook_id': notebook_id, 'workspace': workspace}
                })

            results['suggested_next_actions'] = suggestions
            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Validation error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    async def optimize_notebook(
        self,
        notebook_id: str,
        workspace: str,
        goal: str = "",
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Optimize notebook performance.

        Args:
            notebook_id: Notebook ID
            workspace: Workspace name
            goal: Optimization goal description
            ctx: Context object

        Returns:
            Optimization results and suggestions
        """
        from tools.notebook import (
            analyze_notebook_performance,
            get_notebook_optimization_suggestions
        )

        results = {
            'workflow': 'optimize_notebook',
            'steps_completed': [],
            'notebook_id': notebook_id,
            'goal': goal
        }

        try:
            # Analyze performance
            logger.info(f"[Orchestrator] Analyzing performance for {notebook_id}")
            performance_result = await analyze_notebook_performance(
                workspace=workspace,
                notebook_id=notebook_id,
                ctx=ctx
            )
            performance_data = json.loads(performance_result) if isinstance(performance_result, str) else performance_result

            results['performance'] = performance_data
            results['steps_completed'].append('analyze')

            # Get optimization suggestions
            try:
                suggestions_result = await get_notebook_optimization_suggestions(
                    notebook_id=notebook_id,
                    workspace=workspace,
                    ctx=ctx
                )
                suggestions_data = json.loads(suggestions_result) if isinstance(suggestions_result, str) else suggestions_result

                results['optimization_suggestions'] = suggestions_data
                results['steps_completed'].append('suggest')
            except Exception as e:
                logger.warning(f"[Orchestrator] Could not get optimization suggestions: {e}")

            # Generate next actions
            score = performance_data.get('performance_score', performance_data.get('score', 100))
            if score < 70:
                results['suggested_next_actions'] = [
                    {
                        'tool': 'generate_fabric_code',
                        'priority': 9,
                        'reason': f'Performance score {score}/100 - generate optimized code',
                        'params': {'operation': 'performance_optimization'}
                    }
                ]
            else:
                results['suggested_next_actions'] = [
                    {
                        'tool': 'get_notebook_best_practices',
                        'priority': 6,
                        'reason': 'Performance is good - review best practices',
                        'params': {}
                    }
                ]

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Optimization error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    async def analyze_notebook_comprehensive(
        self,
        notebook_id: str,
        workspace: str,
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Comprehensive notebook analysis.

        Args:
            notebook_id: Notebook ID
            workspace: Workspace name
            ctx: Context object

        Returns:
            Complete analysis results
        """
        from tools.notebook import (
            get_notebook_content,
            validate_pyspark_code,
            analyze_notebook_performance,
            get_notebook_best_practices
        )

        results = {
            'workflow': 'analyze_comprehensive',
            'steps_completed': [],
            'notebook_id': notebook_id
        }

        try:
            # Get content
            content_result = await get_notebook_content(
                workspace=workspace,
                notebook_id=notebook_id,
                ctx=ctx
            )
            content_data = json.loads(content_result) if isinstance(content_result, str) else content_result
            code = self._extract_code_from_notebook(content_data)

            results['content_summary'] = {
                'total_cells': len(content_data.get('cells', [])),
                'code_cells': len([c for c in content_data.get('cells', []) if c.get('type') == 'code']),
            }
            results['steps_completed'].append('get_content')

            # Validate
            validation_result = await validate_pyspark_code(code=code, ctx=ctx)
            validation_data = json.loads(validation_result) if isinstance(validation_result, str) else validation_result
            results['validation'] = validation_data
            results['steps_completed'].append('validate')

            # Performance analysis
            performance_result = await analyze_notebook_performance(
                workspace=workspace,
                notebook_id=notebook_id,
                ctx=ctx
            )
            performance_data = json.loads(performance_result) if isinstance(performance_result, str) else performance_result
            results['performance'] = performance_data
            results['steps_completed'].append('performance')

            # Best practices
            try:
                practices_result = await get_notebook_best_practices(ctx=ctx)
                practices_data = json.loads(practices_result) if isinstance(practices_result, str) else practices_result
                results['best_practices'] = practices_data
                results['steps_completed'].append('best_practices')
            except Exception as e:
                logger.warning(f"[Orchestrator] Could not get best practices: {e}")

            # Generate summary and suggestions
            results['summary'] = {
                'validation_status': 'passed' if not validation_data.get('has_errors') else 'failed',
                'performance_score': performance_data.get('performance_score', performance_data.get('score', 'N/A')),
                'issues_found': len(validation_data.get('issues', [])) + len(performance_data.get('issues', [])),
            }

            results['suggested_next_actions'] = self._suggest_next_steps(
                template_type='unknown',
                validation=validation_data,
                performance=performance_data,
                notebook_id=notebook_id,
                workspace=workspace
            )

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"[Orchestrator] Comprehensive analysis error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    def _extract_code_from_notebook(self, content_data: Dict) -> str:
        """Extract code cells from notebook content."""
        try:
            # Handle different content structures
            if 'cells' in content_data:
                cells = content_data['cells']
            elif 'content' in content_data and isinstance(content_data['content'], dict):
                cells = content_data['content'].get('cells', [])
            else:
                # Try to parse as JSON if it's a string
                if isinstance(content_data, str):
                    parsed = json.loads(content_data)
                    cells = parsed.get('cells', [])
                else:
                    cells = []

            code_cells = []
            for cell in cells:
                if cell.get('type') == 'code' or cell.get('cell_type') == 'code':
                    # Get source code
                    source = cell.get('source', cell.get('content', ''))
                    if isinstance(source, list):
                        source = ''.join(source)
                    if source:
                        code_cells.append(source)

            return '\n\n'.join(code_cells) if code_cells else '# No code found'
        except Exception as e:
            logger.warning(f"[Orchestrator] Error extracting code: {e}")
            return '# Error extracting code'

    def _suggest_next_steps(
        self,
        template_type: str,
        validation: Optional[Dict] = None,
        performance: Optional[Dict] = None,
        notebook_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate context-aware suggestions."""
        suggestions = []

        # Template-specific suggestions
        if template_type == 'etl':
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add data quality checks to ETL pipeline',
                'priority': 8,
                'params': {
                    'operation': 'data_quality',
                    'example': 'Null checks, duplicate detection'
                }
            })
            suggestions.append({
                'tool': 'list_lakehouses',
                'reason': 'Connect ETL to lakehouse for data source',
                'priority': 7,
                'params': {'workspace': workspace} if workspace else {}
            })

        elif template_type == 'ml':
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add feature engineering code',
                'priority': 8,
                'params': {'operation': 'ml_features'}
            })
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add model training code',
                'priority': 8,
                'params': {'operation': 'ml_training'}
            })

        elif template_type == 'streaming':
            suggestions.append({
                'tool': 'generate_fabric_code',
                'reason': 'Setup streaming source configuration',
                'priority': 9,
                'params': {'operation': 'streaming_source'}
            })

        elif template_type == 'analytics':
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add analytical transformations',
                'priority': 7,
                'params': {'operation': 'analytics'}
            })

        # Performance-based suggestions
        if performance:
            score = performance.get('performance_score', performance.get('score', 100))
            if score < 70:
                suggestions.append({
                    'tool': 'generate_fabric_code',
                    'reason': f'Performance score {score}/100 - optimization recommended',
                    'priority': 9,
                    'params': {'operation': 'performance_optimization'}
                })
            elif score < 85:
                suggestions.append({
                    'tool': 'get_notebook_optimization_suggestions',
                    'reason': f'Performance score {score}/100 - room for improvement',
                    'priority': 7,
                    'params': {'notebook_id': notebook_id, 'workspace': workspace} if notebook_id and workspace else {}
                })

        # Validation-based suggestions
        if validation:
            if validation.get('has_errors'):
                suggestions.append({
                    'tool': 'generate_pyspark_code',
                    'reason': 'Fix validation errors',
                    'priority': 10,
                    'params': {'operation': 'fix_errors'}
                })
            elif validation.get('warnings'):
                suggestions.append({
                    'tool': 'get_notebook_best_practices',
                    'reason': 'Address validation warnings',
                    'priority': 6,
                    'params': {}
                })

        # Sort by priority and return top suggestions
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)[:5]


# Singleton instance
notebook_orchestrator = NotebookOrchestrator()
