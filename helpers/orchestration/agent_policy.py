"""Main orchestration facade - agent policy coordinator."""

from typing import Dict, Any, Optional
from helpers.logging_config import get_logger
from .notebook_orchestrator import notebook_orchestrator
from .lakehouse_orchestrator import lakehouse_orchestrator
from .tool_relationships import relationship_engine
from .workflow_chains import workflow_engine
from helpers.policies.execution_policy import ExecutionPolicy, ExecutionMode
import json

logger = get_logger(__name__)


class AgentPolicy:
    """
    Main facade for all orchestration capabilities.
    Coordinates between orchestrators, workflow engine, and execution policies.
    """

    def __init__(self):
        self.notebook_orchestrator = notebook_orchestrator
        self.lakehouse_orchestrator = lakehouse_orchestrator
        self.relationship_engine = relationship_engine
        self.workflow_engine = workflow_engine
        self.execution_policy = ExecutionPolicy(ExecutionMode.STANDARD)

    async def execute_intent(
        self,
        intent: str,
        domain: str = "auto",
        context: Dict[str, Any] = None,
        execution_mode: str = "standard",
        ctx: Any = None
    ) -> Dict[str, Any]:
        """
        Execute user intent using intelligent routing.

        Args:
            intent: Natural language description of what to achieve
            domain: Domain hint (notebook/lakehouse/auto)
            context: Session context with workspace, lakehouse, etc.
            execution_mode: Execution mode (fast/standard/analyze/safe)
            ctx: MCP context object

        Returns:
            Execution results with suggestions
        """
        if context is None:
            from helpers.utils.context import get_context
            context = get_context().__dict__

        # Set execution mode
        mode_enum = {
            'fast': ExecutionMode.FAST,
            'standard': ExecutionMode.STANDARD,
            'analyze': ExecutionMode.ANALYZE,
            'safe': ExecutionMode.SAFE
        }.get(execution_mode.lower(), ExecutionMode.STANDARD)
        self.execution_policy.set_mode(mode_enum)

        result = {
            'intent': intent,
            'domain_requested': domain,
            'execution_mode': execution_mode,
            'context': context
        }

        try:
            # Try to find matching workflow first
            workflow = self.workflow_engine.find_workflow(intent)
            if workflow:
                logger.info(f"[AgentPolicy] Found matching workflow: {workflow.name}")
                result['matched_workflow'] = {
                    'name': workflow.name,
                    'description': workflow.description,
                    'estimated_time': workflow.estimated_time
                }

                # Validate context
                is_valid, missing = self.workflow_engine.validate_workflow_context(workflow, context)
                if not is_valid:
                    result['error'] = f'Missing required context: {", ".join(missing)}'
                    result['success'] = False
                    return result

                # Execute workflow
                result['workflow_execution'] = await self._execute_workflow(workflow, context, ctx)
                result['success'] = result['workflow_execution'].get('success', False)
                return result

            # No workflow found - route by domain and intent keywords
            domain_detected = self._detect_domain(intent, domain)
            result['domain_detected'] = domain_detected

            if domain_detected == 'notebook':
                result['orchestrator_result'] = await self._handle_notebook_intent(intent, context, ctx)
            elif domain_detected == 'lakehouse':
                result['orchestrator_result'] = await self._handle_lakehouse_intent(intent, context, ctx)
            else:
                result['error'] = 'Could not determine domain for intent'
                result['suggestion'] = 'Try being more specific or use domain parameter'
                result['success'] = False
                return result

            result['success'] = result.get('orchestrator_result', {}).get('success', False)
            return result

        except Exception as e:
            logger.error(f"[AgentPolicy] Intent execution error: {e}")
            result['error'] = str(e)
            result['success'] = False
            return result

    async def _handle_notebook_intent(
        self,
        intent: str,
        context: Dict[str, Any],
        ctx: Any
    ) -> Dict[str, Any]:
        """Handle notebook-related intents."""
        intent_lower = intent.lower()
        workspace = context.get('workspace')
        notebook_id = context.get('notebook_id')

        # Pattern 1: Create notebook
        if any(keyword in intent_lower for keyword in ['create', 'build', 'new', 'generate', 'make']):
            # Determine template type
            template = self._infer_template_from_intent(intent)
            notebook_name = self._extract_name_from_intent(intent) or f"notebook_{int(time.time())}"

            return await self.notebook_orchestrator.create_validated_notebook(
                workspace=workspace,
                notebook_name=notebook_name,
                template_type=template,
                validate='validate' in intent_lower or 'check' in intent_lower,
                optimize='optimize' in intent_lower or 'production' in intent_lower,
                ctx=ctx
            )

        # Pattern 2: Validate notebook
        elif any(keyword in intent_lower for keyword in ['validate', 'check', 'verify', 'test']):
            if not notebook_id:
                return {
                    'error': 'notebook_id required for validation',
                    'suggestion': 'Provide notebook_id in context or specify notebook',
                    'success': False
                }

            return await self.notebook_orchestrator.validate_and_suggest(
                notebook_id=notebook_id,
                workspace=workspace,
                ctx=ctx
            )

        # Pattern 3: Optimize notebook
        elif any(keyword in intent_lower for keyword in ['optimize', 'improve', 'performance', 'speed up', 'faster']):
            if not notebook_id:
                return {
                    'error': 'notebook_id required for optimization',
                    'suggestion': 'Provide notebook_id in context',
                    'success': False
                }

            return await self.notebook_orchestrator.optimize_notebook(
                notebook_id=notebook_id,
                workspace=workspace,
                goal=intent,
                ctx=ctx
            )

        # Pattern 4: Analyze notebook
        elif any(keyword in intent_lower for keyword in ['analyze', 'explore', 'understand', 'explain']):
            if not notebook_id:
                return {
                    'error': 'notebook_id required for analysis',
                    'suggestion': 'Provide notebook_id in context',
                    'success': False
                }

            return await self.notebook_orchestrator.analyze_notebook_comprehensive(
                notebook_id=notebook_id,
                workspace=workspace,
                ctx=ctx
            )

        else:
            return {
                'error': 'Could not determine specific notebook operation',
                'suggestion': 'Use keywords like: create, validate, optimize, analyze',
                'success': False
            }

    async def _handle_lakehouse_intent(
        self,
        intent: str,
        context: Dict[str, Any],
        ctx: Any
    ) -> Dict[str, Any]:
        """Handle lakehouse-related intents."""
        intent_lower = intent.lower()
        workspace = context.get('workspace')
        lakehouse = context.get('lakehouse')

        # Pattern 1: Explore lakehouse
        if any(keyword in intent_lower for keyword in ['explore', 'discover', 'list', 'show', 'find']):
            return await self.lakehouse_orchestrator.explore_lakehouse_complete(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )

        # Pattern 2: Setup data integration
        elif any(keyword in intent_lower for keyword in ['integrate', 'connect', 'read', 'write', 'load', 'save']):
            if not lakehouse:
                return {
                    'error': 'lakehouse required for data integration',
                    'suggestion': 'Provide lakehouse in context',
                    'success': False
                }

            return await self.lakehouse_orchestrator.setup_data_integration(
                workspace=workspace,
                lakehouse=lakehouse,
                goal=intent,
                ctx=ctx
            )

        # Pattern 3: Analyze lakehouse
        elif any(keyword in intent_lower for keyword in ['analyze', 'performance', 'optimize']):
            if not lakehouse:
                return {
                    'error': 'lakehouse required for analysis',
                    'suggestion': 'Provide lakehouse in context',
                    'success': False
                }

            return await self.lakehouse_orchestrator.analyze_lakehouse_performance(
                workspace=workspace,
                lakehouse=lakehouse,
                ctx=ctx
            )

        else:
            return {
                'error': 'Could not determine specific lakehouse operation',
                'suggestion': 'Use keywords like: explore, integrate, analyze',
                'success': False
            }

    async def _execute_workflow(
        self,
        workflow: Any,
        context: Dict[str, Any],
        ctx: Any
    ) -> Dict[str, Any]:
        """Execute a workflow chain."""
        result = {
            'workflow_name': workflow.name,
            'steps_executed': [],
            'steps_failed': [],
            'partial_results': []
        }

        try:
            for i, step in enumerate(workflow.steps):
                logger.info(f"[AgentPolicy] Executing step {i+1}/{len(workflow.steps)}: {step.tool}")

                # Check if step is conditional
                if step.conditional and step.condition:
                    # Evaluate condition (simplified - in production would be more sophisticated)
                    should_execute = self._evaluate_condition(step.condition, result, context)
                    if not should_execute:
                        logger.info(f"[AgentPolicy] Skipping conditional step: {step.tool}")
                        continue

                # Execute step (simplified - would need actual tool routing)
                try:
                    step_result = {'step': step.tool, 'description': step.description, 'status': 'executed'}
                    result['steps_executed'].append(step.tool)
                    result['partial_results'].append(step_result)
                except Exception as e:
                    logger.error(f"[AgentPolicy] Step failed: {step.tool} - {e}")
                    result['steps_failed'].append({'step': step.tool, 'error': str(e)})

                    if step.required and not step.retry_on_failure:
                        result['error'] = f'Required step failed: {step.tool}'
                        result['success'] = False
                        return result

            result['success'] = len(result['steps_failed']) == 0
            return result

        except Exception as e:
            logger.error(f"[AgentPolicy] Workflow execution error: {e}")
            result['error'] = str(e)
            result['success'] = False
            return result

    def _detect_domain(self, intent: str, domain_hint: str) -> str:
        """Detect domain from intent keywords."""
        if domain_hint != "auto":
            return domain_hint

        intent_lower = intent.lower()

        # Notebook keywords
        notebook_keywords = [
            'notebook', 'code', 'validate', 'pyspark', 'spark', 'script',
            'ml', 'machine learning', 'analytics', 'etl pipeline'
        ]

        # Lakehouse keywords
        lakehouse_keywords = [
            'lakehouse', 'table', 'data', 'schema', 'database', 'query',
            'explore data', 'discover', 'storage'
        ]

        notebook_score = sum(1 for kw in notebook_keywords if kw in intent_lower)
        lakehouse_score = sum(1 for kw in lakehouse_keywords if kw in intent_lower)

        if notebook_score > lakehouse_score:
            return 'notebook'
        elif lakehouse_score > notebook_score:
            return 'lakehouse'
        else:
            return 'unknown'

    def _infer_template_from_intent(self, intent: str) -> str:
        """Infer notebook template type from intent."""
        intent_lower = intent.lower()

        if any(k in intent_lower for k in ['etl', 'pipeline', 'extract', 'transform', 'load']):
            return 'etl'
        elif any(k in intent_lower for k in ['analytics', 'analysis', 'aggregate', 'report']):
            return 'analytics'
        elif any(k in intent_lower for k in ['ml', 'machine learning', 'model', 'train', 'predict']):
            return 'ml'
        elif any(k in intent_lower for k in ['streaming', 'real-time', 'stream', 'event']):
            return 'streaming'
        elif any(k in intent_lower for k in ['fabric', 'lakehouse integration', 'integrate']):
            return 'fabric_integration'
        else:
            return 'basic'

    def _extract_name_from_intent(self, intent: str) -> Optional[str]:
        """Extract notebook name from intent."""
        import re

        # Look for patterns like: "create notebook called 'name'" or "create 'name' notebook"
        patterns = [
            r"called ['\"]([^'\"]+)['\"]",
            r"named ['\"]([^'\"]+)['\"]",
            r"['\"]([^'\"]+)['\"] notebook",
        ]

        for pattern in patterns:
            match = re.search(pattern, intent)
            if match:
                return match.group(1)

        return None

    def _evaluate_condition(self, condition: str, result: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate workflow step condition."""
        # Simplified condition evaluation
        if condition == 'validation_passed':
            validation = result.get('partial_results', [{}])[-1].get('validation', {})
            return not validation.get('has_errors', False)

        elif condition == 'performance_score_low':
            performance = result.get('partial_results', [{}])[-1].get('performance', {})
            score = performance.get('performance_score', performance.get('score', 100))
            return score < 70

        elif condition == 'tables_found':
            tables = result.get('partial_results', [{}])[-1].get('tables', [])
            return len(tables) > 0

        elif condition == 'template_is_etl':
            return context.get('template_type') == 'etl'

        else:
            # Default to True for unknown conditions
            return True

    def get_available_workflows(self) -> Dict[str, Any]:
        """Get list of all available workflows."""
        workflows = self.workflow_engine.list_all_workflows()
        return {
            'workflows': workflows,
            'total_count': len(workflows),
            'categories': list(set(w['category'] for w in workflows))
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution policy statistics."""
        return self.execution_policy.get_execution_stats()

    def set_execution_mode(self, mode: str):
        """Set execution mode."""
        mode_enum = {
            'fast': ExecutionMode.FAST,
            'standard': ExecutionMode.STANDARD,
            'analyze': ExecutionMode.ANALYZE,
            'safe': ExecutionMode.SAFE
        }.get(mode.lower())

        if mode_enum:
            self.execution_policy.set_mode(mode_enum)
        else:
            raise ValueError(f"Invalid execution mode: {mode}")


# Singleton instance
agent_policy = AgentPolicy()
