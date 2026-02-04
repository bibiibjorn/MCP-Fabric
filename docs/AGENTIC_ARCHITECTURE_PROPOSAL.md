# Agentic Architecture Proposal for Fabric MCP Server

## Executive Summary

This document proposes implementing sophisticated agentic logic patterns in the Fabric MCP server, inspired by the proven architecture of the MCP-PowerBi-Finvision server. The goal is to enable intelligent, autonomous multi-step operations while maintaining deterministic behavior and user control.

## Current Architecture Analysis

### Strengths
- ✅ Comprehensive PySpark notebook tools (13 categories)
- ✅ Rich template system (6 templates)
- ✅ Code validation and analysis capabilities
- ✅ Public API documentation access
- ✅ Context management system

### Gaps Identified
- ❌ No orchestration layer for complex workflows
- ❌ Limited tool-to-tool coordination
- ❌ No intelligent suggestion engine
- ❌ Manual multi-step operations
- ❌ No workflow templates for common patterns
- ❌ Limited decision-making logic

## Proposed Agentic Patterns

### Pattern 1: Orchestration Layer

**Create**: `helpers/orchestration/` directory structure

```
helpers/orchestration/
├── __init__.py
├── agent_policy.py          # Main facade
├── notebook_orchestrator.py # Notebook workflows
├── lakehouse_orchestrator.py # Data workflows
├── pipeline_orchestrator.py  # ETL workflows
└── analysis_orchestrator.py  # Performance analysis
```

**Implementation Example**:
```python
# notebook_orchestrator.py
class NotebookOrchestrator:
    """Orchestrates complex notebook development workflows."""

    async def create_validated_notebook(
        self,
        workspace: str,
        notebook_name: str,
        template_type: str,
        validate: bool = True,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Multi-step notebook creation with validation and optimization.

        Workflow:
        1. Generate notebook from template
        2. Validate PySpark and Fabric compatibility
        3. Optimize code if requested
        4. Deploy to Fabric
        5. Return deployment status + suggestions
        """
        results = {}

        # Step 1: Create notebook
        notebook_id = await self._create_notebook(
            workspace, notebook_name, template_type
        )
        results['created'] = True
        results['notebook_id'] = notebook_id

        # Step 2: Validate (conditional)
        if validate:
            validation = await self._validate_notebook(notebook_id)
            results['validation'] = validation

            if validation['has_errors']:
                results['suggested_next_actions'] = [
                    {
                        'tool': 'validate_pyspark_code',
                        'reason': 'Fix syntax errors before deployment',
                        'priority': 10
                    }
                ]
                return results

        # Step 3: Optimize (conditional)
        if optimize:
            optimization = await self._optimize_notebook(notebook_id)
            results['optimization'] = optimization

        # Step 4: Deploy
        deployment = await self._deploy_notebook(workspace, notebook_id)
        results['deployment'] = deployment

        # Step 5: Add suggestions
        results['suggested_next_actions'] = self._suggest_next_steps(
            notebook_id, template_type, validation
        )

        return results

    def _suggest_next_steps(self, notebook_id, template, validation):
        """Context-aware suggestions for next actions."""
        suggestions = []

        if template == 'etl':
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add data quality checks to ETL pipeline',
                'priority': 8,
                'example': 'operation=data_quality, table=source_table'
            })

        if template == 'ml':
            suggestions.append({
                'tool': 'analyze_notebook_performance',
                'reason': 'Benchmark ML pipeline performance',
                'priority': 7
            })

        if validation and validation['performance_score'] < 70:
            suggestions.append({
                'tool': 'generate_fabric_code',
                'reason': 'Performance score below 70, consider optimizations',
                'priority': 9,
                'example': 'operation=performance_optimization'
            })

        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)
```

### Pattern 2: Tool Relationship Registry

**Create**: `helpers/orchestration/tool_relationships.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Callable
from enum import Enum

class RelationType(Enum):
    REQUIRES = "requires"        # Must run before
    SUGGESTS = "suggests"        # Should consider running
    ENRICHES = "enriches"        # Adds value to
    VALIDATES = "validates"      # Checks result of
    INVERSE = "inverse"          # Opposite operation

@dataclass
class ToolRelationship:
    """Defines relationship between two tools."""
    source_tool: str
    related_tool: str
    relationship_type: RelationType
    condition: Callable[[Dict], bool]  # When to suggest
    context_mapping: Dict[str, str]    # Parameter mapping
    priority: int                       # 1-10, higher = more important
    reason: str                        # Why suggest this tool

# Define tool relationships
TOOL_RELATIONSHIPS = [
    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('created', False),
        context_mapping={'notebook_id': 'code_source'},
        priority=9,
        reason="Validate new notebook for syntax and best practices"
    ),
    ToolRelationship(
        source_tool="validate_pyspark_code",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('has_issues', False),
        context_mapping={},
        priority=8,
        reason="Generate corrected code for identified issues"
    ),
    ToolRelationship(
        source_tool="list_tables",
        related_tool="get_all_lakehouse_schemas",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('tables', [])) > 0,
        context_mapping={'lakehouse': 'lakehouse', 'workspace': 'workspace'},
        priority=7,
        reason="Get detailed schemas for all tables found"
    ),
    ToolRelationship(
        source_tool="get_lakehouse_table_schema",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('columns', []),
        context_mapping={'table_name': 'source_table'},
        priority=8,
        reason="Generate PySpark code to work with this table"
    ),
    ToolRelationship(
        source_tool="analyze_notebook_performance",
        related_tool="generate_fabric_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('score', 100) < 70,
        context_mapping={},
        priority=9,
        reason="Low performance score - generate optimized code"
    ),
    ToolRelationship(
        source_tool="create_lakehouse",
        related_tool="create_pyspark_notebook",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('created', False),
        context_mapping={'lakehouse': 'lakehouse_name', 'workspace': 'workspace'},
        priority=7,
        reason="Create ETL notebook for new lakehouse"
    ),
    # Add 40+ more relationships...
]

# Define workflow chains
@dataclass
class WorkflowChain:
    """Pre-defined multi-step workflow."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    trigger_conditions: List[str]
    estimated_time: str

WORKFLOW_CHAINS = [
    WorkflowChain(
        name="complete_notebook_development",
        description="Full notebook development lifecycle",
        steps=[
            {
                'tool': 'create_pyspark_notebook',
                'params': {'template_type': 'etl'},
                'description': 'Create ETL notebook from template'
            },
            {
                'tool': 'validate_pyspark_code',
                'params': {'code_source': 'notebook_id'},
                'description': 'Validate syntax and best practices'
            },
            {
                'tool': 'generate_pyspark_code',
                'params': {'operation': 'data_quality'},
                'description': 'Add data quality checks',
                'conditional': True
            },
            {
                'tool': 'analyze_notebook_performance',
                'params': {'notebook_id': 'notebook_id'},
                'description': 'Analyze performance and optimization opportunities'
            }
        ],
        trigger_conditions=[
            "user_wants_complete_notebook",
            "create_production_ready_notebook",
            "full_notebook_development"
        ],
        estimated_time="30-60 seconds"
    ),
    WorkflowChain(
        name="lakehouse_data_exploration",
        description="Comprehensive lakehouse and table analysis",
        steps=[
            {
                'tool': 'list_lakehouses',
                'params': {'workspace': 'workspace'},
                'description': 'List all lakehouses'
            },
            {
                'tool': 'list_tables',
                'params': {'lakehouse': 'lakehouse'},
                'description': 'List tables in lakehouse'
            },
            {
                'tool': 'get_all_lakehouse_schemas',
                'params': {'lakehouse': 'lakehouse'},
                'description': 'Get schemas for all tables'
            },
            {
                'tool': 'generate_pyspark_code',
                'params': {'operation': 'read_table'},
                'description': 'Generate sample reading code'
            }
        ],
        trigger_conditions=[
            "explore_lakehouse",
            "understand_data_structure",
            "data_discovery"
        ],
        estimated_time="15-30 seconds"
    ),
    WorkflowChain(
        name="notebook_optimization_pipeline",
        description="Analyze and optimize existing notebook",
        steps=[
            {
                'tool': 'get_notebook_content',
                'params': {'notebook_id': 'notebook_id'},
                'description': 'Retrieve notebook content'
            },
            {
                'tool': 'validate_fabric_code',
                'params': {'code': 'notebook_code'},
                'description': 'Validate Fabric compatibility'
            },
            {
                'tool': 'analyze_notebook_performance',
                'params': {'notebook_id': 'notebook_id'},
                'description': 'Analyze performance bottlenecks'
            },
            {
                'tool': 'generate_fabric_code',
                'params': {'operation': 'performance_optimization'},
                'description': 'Generate optimized code',
                'conditional': True
            }
        ],
        trigger_conditions=[
            "optimize_notebook",
            "improve_performance",
            "notebook_slow"
        ],
        estimated_time="20-40 seconds"
    )
]

class ToolRelationshipEngine:
    """Engine for discovering and suggesting related tools."""

    def get_suggested_tools(
        self,
        source_tool: str,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get tools suggested after executing source_tool."""
        suggestions = []

        for relationship in TOOL_RELATIONSHIPS:
            if relationship.source_tool == source_tool:
                # Check if condition is met
                if relationship.condition(result):
                    # Map context parameters
                    mapped_params = {
                        target: context.get(source)
                        for source, target in relationship.context_mapping.items()
                        if source in context
                    }

                    suggestions.append({
                        'tool': relationship.related_tool,
                        'priority': relationship.priority,
                        'reason': relationship.reason,
                        'params': mapped_params,
                        'relationship': relationship.relationship_type.value
                    })

        # Sort by priority
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)

    def get_workflow_chain(self, trigger: str) -> WorkflowChain:
        """Get workflow chain matching trigger condition."""
        for chain in WORKFLOW_CHAINS:
            if any(trigger.lower() in condition.lower()
                   for condition in chain.trigger_conditions):
                return chain
        return None
```

### Pattern 3: Intent-Driven Execution

**Add to**: `tools/notebook.py`

```python
@mcp.tool()
async def execute_notebook_intent(
    goal: str,
    workspace: str | None = None,
    notebook_id: str | None = None,
    lakehouse: str | None = None,
    additional_context: str | None = None
) -> str:
    """
    Execute notebook operations based on natural language intent.

    Intelligent routing based on keywords in the goal:
    - Create/build → create_pyspark_notebook
    - Validate/check → validate_pyspark_code
    - Optimize/improve → analyze_notebook_performance + generate_fabric_code
    - Explore/understand → get_notebook_content + analyze
    - Compare/benchmark → performance comparison workflow

    Args:
        goal: Natural language description of what you want to achieve
        workspace: Fabric workspace name (resolved from context if not provided)
        notebook_id: Notebook ID (for existing notebooks)
        lakehouse: Lakehouse name (for data operations)
        additional_context: Extra context for complex intents

    Returns:
        JSON string with execution results and suggested next actions
    """
    ctx = get_context()
    workspace = workspace or ctx.workspace
    goal_lower = goal.lower()

    # Pattern 1: Create notebook
    if any(keyword in goal_lower for keyword in [
        "create", "build", "new", "generate", "make"
    ]):
        # Determine template type from goal
        template = _infer_template_from_goal(goal)
        return await notebook_orchestrator.create_validated_notebook(
            workspace=workspace,
            notebook_name=_extract_name_from_goal(goal),
            template_type=template,
            validate=True,
            optimize="production" in goal_lower
        )

    # Pattern 2: Validate notebook
    if any(keyword in goal_lower for keyword in [
        "validate", "check", "verify", "test", "syntax"
    ]):
        if not notebook_id:
            return json.dumps({
                "error": "notebook_id required for validation",
                "suggestion": "Provide notebook_id or use 'list_notebooks' first"
            })

        return await notebook_orchestrator.validate_and_suggest(
            notebook_id=notebook_id,
            workspace=workspace
        )

    # Pattern 3: Optimize notebook
    if any(keyword in goal_lower for keyword in [
        "optimize", "improve", "performance", "speed up", "faster"
    ]):
        if not notebook_id:
            return json.dumps({
                "error": "notebook_id required for optimization",
                "suggestion": "Provide notebook_id or use 'list_notebooks' first"
            })

        return await notebook_orchestrator.optimize_notebook(
            notebook_id=notebook_id,
            workspace=workspace,
            goal=goal
        )

    # Pattern 4: Explore/analyze notebook
    if any(keyword in goal_lower for keyword in [
        "explore", "understand", "analyze", "explain", "what does"
    ]):
        if not notebook_id:
            # List all notebooks for exploration
            return await notebook_orchestrator.explore_notebooks(
                workspace=workspace
            )

        return await notebook_orchestrator.analyze_notebook_comprehensive(
            notebook_id=notebook_id,
            workspace=workspace
        )

    # Pattern 5: Compare notebooks
    if any(keyword in goal_lower for keyword in [
        "compare", "benchmark", "which is better", "difference"
    ]):
        # Extract multiple notebook IDs from context
        notebook_ids = _extract_notebook_ids(goal, additional_context)
        return await notebook_orchestrator.compare_notebooks(
            notebook_ids=notebook_ids,
            workspace=workspace
        )

    # Pattern 6: Data integration
    if any(keyword in goal_lower for keyword in [
        "read from", "write to", "connect to", "lakehouse", "table"
    ]):
        if not lakehouse:
            lakehouse = ctx.lakehouse

        return await notebook_orchestrator.setup_data_integration(
            workspace=workspace,
            lakehouse=lakehouse,
            goal=goal
        )

    # Fallback: Plan the operation
    return await notebook_orchestrator.plan_notebook_operation(
        goal=goal,
        workspace=workspace,
        notebook_id=notebook_id,
        context=ctx
    )

def _infer_template_from_goal(goal: str) -> str:
    """Infer template type from goal description."""
    goal_lower = goal.lower()

    if any(k in goal_lower for k in ["etl", "pipeline", "extract", "transform"]):
        return "etl"
    elif any(k in goal_lower for k in ["analytics", "analysis", "aggregate"]):
        return "analytics"
    elif any(k in goal_lower for k in ["ml", "machine learning", "model", "train"]):
        return "ml"
    elif any(k in goal_lower for k in ["streaming", "real-time", "stream"]):
        return "streaming"
    elif any(k in goal_lower for k in ["fabric", "lakehouse integration"]):
        return "fabric_integration"
    else:
        return "basic"
```

### Pattern 4: Enhanced Response Format with Suggestions

**Modify all tools to return**:

```python
def _format_tool_response(
    success: bool,
    data: Any,
    tool_name: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Standard response format with suggestions."""

    response = {
        'success': success,
        'data': data,
        'tool': tool_name,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Add intelligent suggestions
    relationship_engine = ToolRelationshipEngine()
    suggestions = relationship_engine.get_suggested_tools(
        source_tool=tool_name,
        result=data,
        context=context
    )

    if suggestions:
        response['suggested_next_actions'] = suggestions[:5]  # Top 5

    # Add workflow recommendations
    if context.get('enable_workflows'):
        workflow = relationship_engine.get_workflow_chain(tool_name)
        if workflow:
            response['available_workflow'] = {
                'name': workflow.name,
                'description': workflow.description,
                'estimated_time': workflow.estimated_time,
                'steps': [s['description'] for s in workflow.steps]
            }

    return response
```

### Pattern 5: Query/Operation Policy System

**Create**: `helpers/policies/execution_policy.py`

```python
from enum import Enum
from typing import Any, Dict, Callable

class ExecutionMode(Enum):
    FAST = "fast"           # Quick preview, minimal validation
    STANDARD = "standard"   # Normal execution with validation
    ANALYZE = "analyze"     # Full analysis with performance metrics
    SAFE = "safe"          # Maximum validation, rollback capability

class ExecutionPolicy:
    """Determines execution strategy for operations."""

    def __init__(self, mode: ExecutionMode = ExecutionMode.STANDARD):
        self.mode = mode

    async def execute_notebook_operation(
        self,
        operation: Callable,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute notebook operation with policy-based controls."""

        if self.mode == ExecutionMode.FAST:
            # Skip validation, preview only
            params['validate'] = False
            params['limit'] = 10
            result = await operation(**params)

        elif self.mode == ExecutionMode.SAFE:
            # Maximum validation, create backup
            await self._create_backup(context.get('notebook_id'))
            params['validate'] = True
            params['dry_run'] = True

            # Validate first
            validation = await self._validate_operation(operation, params)
            if not validation['safe']:
                return {
                    'error': 'Operation failed safety check',
                    'details': validation['issues'],
                    'suggestion': 'Review issues or use STANDARD mode'
                }

            # Execute with rollback capability
            result = await self._execute_with_rollback(operation, params)

        elif self.mode == ExecutionMode.ANALYZE:
            # Full execution with performance analysis
            start_time = time.time()
            result = await operation(**params)
            execution_time = time.time() - start_time

            # Add performance metrics
            result['performance_metrics'] = {
                'execution_time': execution_time,
                'analysis': await self._analyze_performance(result)
            }

        else:  # STANDARD
            result = await operation(**params)

        return result

    async def _validate_operation(
        self,
        operation: Callable,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pre-execution validation."""
        issues = []

        # Check required context
        if 'workspace' not in params:
            issues.append('Missing workspace context')

        # Check permissions (if available)
        # Check resource availability
        # Check syntax (for code operations)

        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. Create orchestration directory structure
2. Implement NotebookOrchestrator with 3 core workflows
3. Define initial ToolRelationshipRegistry (20 relationships)
4. Add ExecutionPolicy system

### Phase 2: Enhanced Intelligence (Week 3-4)
1. Implement ToolRelationshipEngine
2. Add 3 WorkflowChains for common patterns
3. Implement intent-driven execution for notebooks
4. Update all tools to include suggestions

### Phase 3: Advanced Capabilities (Week 5-6)
1. Add LakehouseOrchestrator and PipelineOrchestrator
2. Expand ToolRelationshipRegistry to 50+ relationships
3. Implement 5 additional WorkflowChains
4. Add workflow templates UI/documentation

### Phase 4: Optimization & Testing (Week 7-8)
1. Performance optimization and lazy loading
2. Comprehensive testing suite
3. Documentation and examples
4. User acceptance testing

## Success Metrics

### Quantitative
- ⏱️ **Reduced multi-step operations**: 60% fewer manual tool invocations
- 🎯 **Suggestion accuracy**: 80%+ of suggestions are followed
- ⚡ **Workflow execution time**: 40% faster than manual steps
- 📈 **Tool usage**: 2x increase in advanced tool adoption

### Qualitative
- 🧠 **Intelligent behavior**: LLM can autonomously complete complex tasks
- 🔄 **Seamless workflows**: Natural transitions between related operations
- 💡 **Proactive assistance**: Relevant suggestions without explicit requests
- 🛡️ **Safety**: No unintended operations, clear user control

## Risk Mitigation

### Risk: Over-automation
**Mitigation**: Suggestions only, never auto-execute multi-step workflows without explicit consent

### Risk: Performance overhead
**Mitigation**: Lazy loading, optional suggestion engine, performance mode flags

### Risk: Incorrect suggestions
**Mitigation**: Priority scoring, condition validation, user feedback loop

### Risk: Breaking changes
**Mitigation**: Backward compatible, orchestrators wrap existing tools

## Comparison: Current vs. Proposed

| Feature | Current | Proposed |
|---------|---------|----------|
| Multi-step operations | Manual, 5+ tool calls | Orchestrated, 1 call |
| Tool discovery | User must know tools | Intelligent suggestions |
| Workflow execution | Linear, explicit | Branching, adaptive |
| Error handling | Per-tool | Cascading fallbacks |
| Context awareness | Basic (workspace/lakehouse) | Rich (intent, history, patterns) |
| Performance optimization | Manual | Policy-driven, automatic |

## Conclusion

Implementing these agentic patterns will transform the Fabric MCP server from a collection of tools into an intelligent assistant capable of:

1. **Understanding intent** and routing to appropriate workflows
2. **Suggesting logical next steps** based on context
3. **Orchestrating complex operations** with minimal user input
4. **Adapting execution strategy** based on goals and safety requirements
5. **Learning from patterns** to improve suggestions over time

This architecture is proven in the PowerBI Finvision server and can significantly enhance developer productivity while maintaining safety and user control.

---

**Next Steps**: Review this proposal and decide which patterns to implement in Phase 1.
