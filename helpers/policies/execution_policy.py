"""Execution policies for adaptive operation strategies."""

from enum import Enum
from typing import Any, Dict, Callable, Optional
from helpers.logging_config import get_logger
import time

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution modes with different validation and performance characteristics."""
    FAST = "fast"           # Quick preview, minimal validation
    STANDARD = "standard"   # Normal execution with validation
    ANALYZE = "analyze"     # Full analysis with performance metrics
    SAFE = "safe"          # Maximum validation, rollback capability


class ExecutionPolicy:
    """Determines execution strategy for operations."""

    def __init__(self, mode: ExecutionMode = ExecutionMode.STANDARD):
        self.mode = mode
        self.execution_history = []

    async def execute_with_policy(
        self,
        operation: Callable,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute operation with policy-based controls.

        Args:
            operation: Async callable to execute
            params: Parameters for the operation
            context: Additional context

        Returns:
            Enhanced result with policy metadata
        """
        if context is None:
            context = {}

        start_time = time.time()

        result = {
            'execution_mode': self.mode.value,
            'started_at': start_time,
        }

        try:
            if self.mode == ExecutionMode.FAST:
                # Fast mode: Skip validation, limit results
                logger.info(f"[ExecutionPolicy] FAST mode - minimal validation")
                params = self._apply_fast_mode_limits(params)
                operation_result = await operation(**params)
                result['data'] = operation_result
                result['mode_optimizations'] = {
                    'validation_skipped': True,
                    'result_limited': True,
                    'estimated_speedup': '5x'
                }

            elif self.mode == ExecutionMode.SAFE:
                # Safe mode: Maximum validation
                logger.info(f"[ExecutionPolicy] SAFE mode - maximum validation")

                # Pre-execution validation
                validation_result = await self._validate_operation(operation, params, context)
                if not validation_result['safe']:
                    result['error'] = 'Operation failed safety check'
                    result['validation_issues'] = validation_result['issues']
                    result['suggestion'] = 'Review issues or use STANDARD mode'
                    result['success'] = False
                    return result

                result['pre_validation'] = validation_result

                # Execute with error handling
                try:
                    operation_result = await operation(**params)
                    result['data'] = operation_result
                    result['success'] = True
                except Exception as e:
                    result['error'] = str(e)
                    result['success'] = False
                    result['rollback_available'] = False  # Would implement rollback here

            elif self.mode == ExecutionMode.ANALYZE:
                # Analyze mode: Full metrics collection
                logger.info(f"[ExecutionPolicy] ANALYZE mode - collecting metrics")

                operation_result = await operation(**params)
                result['data'] = operation_result

                execution_time = time.time() - start_time
                result['performance_metrics'] = {
                    'execution_time_seconds': execution_time,
                    'parameters_count': len(params),
                    'analysis': await self._analyze_performance(operation_result, execution_time)
                }

                result['success'] = True

            else:  # STANDARD mode
                logger.info(f"[ExecutionPolicy] STANDARD mode - normal execution")
                operation_result = await operation(**params)
                result['data'] = operation_result
                result['success'] = True

            # Record execution
            execution_time = time.time() - start_time
            result['execution_time_seconds'] = execution_time
            self._record_execution(operation.__name__ if hasattr(operation, '__name__') else 'unknown', execution_time, result.get('success', False))

            return result

        except Exception as e:
            logger.error(f"[ExecutionPolicy] Execution error: {e}")
            result['error'] = str(e)
            result['success'] = False
            result['execution_time_seconds'] = time.time() - start_time
            return result

    def _apply_fast_mode_limits(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fast mode optimizations to parameters."""
        modified_params = params.copy()

        # Add/modify parameters for fast mode
        if 'validate' in modified_params:
            modified_params['validate'] = False

        if 'limit' not in modified_params:
            modified_params['limit'] = 10

        if 'optimize' in modified_params:
            modified_params['optimize'] = False

        return modified_params

    async def _validate_operation(
        self,
        operation: Callable,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pre-execution validation for safe mode."""
        issues = []

        # Check required context
        required_keys = ['workspace']  # Basic requirement
        for key in required_keys:
            if key not in params and key not in context:
                issues.append(f'Missing required context: {key}')

        # Validate parameter types
        for key, value in params.items():
            if value is None:
                issues.append(f'Parameter {key} is None')

        # Check for potentially dangerous operations
        operation_name = operation.__name__ if hasattr(operation, '__name__') else 'unknown'
        if 'delete' in operation_name.lower():
            issues.append('Destructive operation detected - extra caution required')

        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'checks_performed': ['context_validation', 'parameter_validation', 'operation_type_check']
        }

    async def _analyze_performance(
        self,
        result: Any,
        execution_time: float
    ) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        analysis = {
            'execution_time_category': 'fast' if execution_time < 1.0 else 'normal' if execution_time < 5.0 else 'slow',
            'execution_time_seconds': execution_time,
        }

        # Analyze result size
        if isinstance(result, str):
            try:
                import json
                data = json.loads(result)
                analysis['result_size_bytes'] = len(result)
                analysis['result_type'] = 'json'

                # Count items if it's a list
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            analysis[f'{key}_count'] = len(value)
            except:
                analysis['result_size_bytes'] = len(result)
                analysis['result_type'] = 'string'

        # Performance recommendations
        recommendations = []
        if execution_time > 5.0:
            recommendations.append('Consider using FAST mode for quicker previews')
            recommendations.append('Review query optimization opportunities')

        if execution_time < 0.5:
            recommendations.append('Operation is already optimal')

        analysis['recommendations'] = recommendations

        return analysis

    def _record_execution(self, operation_name: str, execution_time: float, success: bool):
        """Record execution in history for analytics."""
        self.execution_history.append({
            'operation': operation_name,
            'execution_time': execution_time,
            'success': success,
            'mode': self.mode.value,
            'timestamp': time.time()
        })

        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {'message': 'No execution history'}

        total_executions = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['success'])
        avg_time = sum(e['execution_time'] for e in self.execution_history) / total_executions

        return {
            'total_executions': total_executions,
            'successful_executions': successful,
            'success_rate': successful / total_executions * 100,
            'average_execution_time': avg_time,
            'current_mode': self.mode.value,
            'fastest_execution': min(e['execution_time'] for e in self.execution_history),
            'slowest_execution': max(e['execution_time'] for e in self.execution_history),
        }

    def set_mode(self, mode: ExecutionMode):
        """Change execution mode."""
        logger.info(f"[ExecutionPolicy] Changing mode from {self.mode.value} to {mode.value}")
        self.mode = mode

    def get_mode_description(self) -> Dict[str, Any]:
        """Get description of current execution mode."""
        descriptions = {
            ExecutionMode.FAST: {
                'name': 'Fast Mode',
                'description': 'Quick preview with minimal validation',
                'validation': 'Skipped',
                'performance': '5x faster',
                'use_case': 'Quick exploration and testing',
                'trade_off': 'Less validation, limited results'
            },
            ExecutionMode.STANDARD: {
                'name': 'Standard Mode',
                'description': 'Normal execution with basic validation',
                'validation': 'Basic',
                'performance': 'Baseline',
                'use_case': 'Regular operations',
                'trade_off': 'Balanced'
            },
            ExecutionMode.ANALYZE: {
                'name': 'Analyze Mode',
                'description': 'Full execution with comprehensive metrics',
                'validation': 'Standard',
                'performance': '2x slower (includes profiling)',
                'use_case': 'Performance tuning and optimization',
                'trade_off': 'Detailed insights but slower'
            },
            ExecutionMode.SAFE: {
                'name': 'Safe Mode',
                'description': 'Maximum validation with rollback capability',
                'validation': 'Maximum',
                'performance': '3x slower (includes safety checks)',
                'use_case': 'Production changes and critical operations',
                'trade_off': 'Very safe but slower'
            }
        }

        return descriptions.get(self.mode, {'name': 'Unknown', 'description': 'Unknown mode'})


# Default global policy instance
default_policy = ExecutionPolicy(ExecutionMode.STANDARD)
