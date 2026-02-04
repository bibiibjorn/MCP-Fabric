# Quick Start: Implementing Agentic Logic in Fabric MCP Server

## Phase 1: Foundation (Week 1-2)

### Step 1: Create Directory Structure

```bash
mkdir -p helpers/orchestration
mkdir -p helpers/policies
touch helpers/orchestration/__init__.py
touch helpers/orchestration/agent_policy.py
touch helpers/orchestration/notebook_orchestrator.py
touch helpers/orchestration/tool_relationships.py
touch helpers/policies/__init__.py
touch helpers/policies/execution_policy.py
```

### Step 2: Implement Basic NotebookOrchestrator

**File**: `helpers/orchestration/notebook_orchestrator.py`

```python
"""Notebook orchestration for multi-step workflows."""

from helpers.utils.context import get_context
from helpers.logging_config import get_logger
from typing import Dict, Any
import json

logger = get_logger(__name__)


class NotebookOrchestrator:
    """Orchestrates complex notebook development workflows."""

    async def create_validated_notebook(
        self,
        workspace: str,
        notebook_name: str,
        template_type: str,
        validate: bool = True,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Create notebook with automatic validation and optimization.

        Args:
            workspace: Fabric workspace name
            notebook_name: Name for new notebook
            template_type: Template (basic/etl/analytics/ml/fabric_integration/streaming)
            validate: Run validation after creation
            optimize: Run optimization analysis

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
            'steps_completed': []
        }

        try:
            # Step 1: Create notebook
            logger.info(f"Creating notebook: {notebook_name}")
            creation_result = await create_pyspark_notebook(
                workspace=workspace,
                notebook_name=notebook_name,
                template_type=template_type
            )
            results['notebook_created'] = creation_result
            results['steps_completed'].append('create')

            # Extract notebook ID from result
            # (Assuming creation returns JSON with notebook_id)
            creation_data = json.loads(creation_result)
            notebook_id = creation_data.get('notebook_id')

            if not notebook_id:
                results['error'] = 'Failed to get notebook_id from creation'
                return results

            # Step 2: Validate (conditional)
            if validate:
                logger.info(f"Validating notebook: {notebook_id}")
                # Get notebook content first
                from tools.notebook import get_notebook_content
                content_result = await get_notebook_content(
                    workspace=workspace,
                    notebook_id=notebook_id
                )
                content_data = json.loads(content_result)

                # Extract code cells
                code = self._extract_code_from_notebook(content_data)

                # Validate
                validation_result = await validate_pyspark_code(code=code)
                results['validation'] = json.loads(validation_result)
                results['steps_completed'].append('validate')

                # Check for critical errors
                validation_data = results['validation']
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
                    # Don't proceed to optimization if errors exist
                    return results

            # Step 3: Optimize (conditional)
            if optimize:
                logger.info(f"Analyzing performance: {notebook_id}")
                performance_result = await analyze_notebook_performance(
                    workspace=workspace,
                    notebook_id=notebook_id
                )
                results['performance'] = json.loads(performance_result)
                results['steps_completed'].append('optimize')

            # Step 4: Add intelligent suggestions
            results['suggested_next_actions'] = self._suggest_next_steps(
                template_type=template_type,
                validation=results.get('validation'),
                performance=results.get('performance')
            )

            results['success'] = True
            return results

        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    def _extract_code_from_notebook(self, content_data: Dict) -> str:
        """Extract code cells from notebook content."""
        # Implementation depends on notebook structure
        # Placeholder:
        cells = content_data.get('cells', [])
        code_cells = [
            cell.get('content', '')
            for cell in cells
            if cell.get('type') == 'code'
        ]
        return '\n\n'.join(code_cells)

    def _suggest_next_steps(
        self,
        template_type: str,
        validation: Dict = None,
        performance: Dict = None
    ) -> list:
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
                'params': {}
            })

        elif template_type == 'ml':
            suggestions.append({
                'tool': 'generate_pyspark_code',
                'reason': 'Add feature engineering code',
                'priority': 8,
                'params': {'operation': 'ml_features'}
            })

        elif template_type == 'streaming':
            suggestions.append({
                'tool': 'generate_fabric_code',
                'reason': 'Setup streaming source configuration',
                'priority': 9,
                'params': {'operation': 'streaming_source'}
            })

        # Performance-based suggestions
        if performance:
            score = performance.get('performance_score', 100)
            if score < 70:
                suggestions.append({
                    'tool': 'generate_fabric_code',
                    'reason': f'Performance score {score}/100 - optimization recommended',
                    'priority': 9,
                    'params': {'operation': 'performance_optimization'}
                })

        # Sort by priority
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)


# Singleton instance
notebook_orchestrator = NotebookOrchestrator()
```

### Step 3: Create Basic Tool Relationships

**File**: `helpers/orchestration/tool_relationships.py`

```python
"""Tool relationship definitions for intelligent suggestions."""

from dataclasses import dataclass
from typing import List, Dict, Callable, Any
from enum import Enum


class RelationType(Enum):
    """Types of relationships between tools."""
    REQUIRES = "requires"     # Must run before
    SUGGESTS = "suggests"     # Should consider running
    ENRICHES = "enriches"     # Adds value to
    VALIDATES = "validates"   # Checks result of


@dataclass
class ToolRelationship:
    """Defines relationship between two tools."""
    source_tool: str
    related_tool: str
    relationship_type: RelationType
    condition: Callable[[Dict], bool]
    priority: int  # 1-10, higher = more important
    reason: str
    context_mapping: Dict[str, str] = None

    def __post_init__(self):
        if self.context_mapping is None:
            self.context_mapping = {}


# Define initial relationships (start with 10, expand to 50+)
TOOL_RELATIONSHIPS = [
    # Notebook creation → validation
    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('success', False),
        priority=9,
        reason="Validate new notebook for syntax and best practices",
        context_mapping={'notebook_id': 'code_source'}
    ),

    # Validation → code generation (if issues found)
    ToolRelationship(
        source_tool="validate_pyspark_code",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('has_issues', False),
        priority=8,
        reason="Generate corrected code for identified issues"
    ),

    # List tables → get schemas
    ToolRelationship(
        source_tool="list_tables",
        related_tool="get_all_lakehouse_schemas",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('tables', [])) > 0,
        priority=7,
        reason="Get detailed schemas for all tables found",
        context_mapping={'lakehouse': 'lakehouse', 'workspace': 'workspace'}
    ),

    # Get schema → generate code
    ToolRelationship(
        source_tool="get_lakehouse_table_schema",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'columns' in result,
        priority=8,
        reason="Generate PySpark code to work with this table",
        context_mapping={'table_name': 'source_table'}
    ),

    # Performance analysis → optimization
    ToolRelationship(
        source_tool="analyze_notebook_performance",
        related_tool="generate_fabric_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('performance_score', 100) < 70,
        priority=9,
        reason="Low performance score - generate optimized code",
        context_mapping={'notebook_id': 'notebook_id'}
    ),

    # Create lakehouse → create notebook
    ToolRelationship(
        source_tool="create_lakehouse",
        related_tool="create_pyspark_notebook",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('created', False),
        priority=7,
        reason="Create ETL notebook for new lakehouse",
        context_mapping={'lakehouse': 'lakehouse_name', 'workspace': 'workspace'}
    ),

    # List lakehouses → list tables
    ToolRelationship(
        source_tool="list_lakehouses",
        related_tool="list_tables",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('lakehouses', [])) > 0,
        priority=6,
        reason="Explore tables in discovered lakehouses",
        context_mapping={'lakehouse': 'first_lakehouse'}
    ),

    # Generate code → validate
    ToolRelationship(
        source_tool="generate_pyspark_code",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.VALIDATES,
        condition=lambda result: 'code' in result,
        priority=7,
        reason="Validate generated code before use"
    ),

    # Notebook created → get content
    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="get_notebook_content",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: result.get('notebook_id'),
        priority=5,
        reason="View notebook content after creation",
        context_mapping={'notebook_id': 'notebook_id'}
    ),

    # List workspaces → list lakehouses
    ToolRelationship(
        source_tool="list_workspaces",
        related_tool="list_lakehouses",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('workspaces', [])) > 0,
        priority=6,
        reason="Explore lakehouses in discovered workspaces",
        context_mapping={'workspace': 'first_workspace'}
    ),
]


class ToolRelationshipEngine:
    """Engine for discovering and suggesting related tools."""

    def get_suggested_tools(
        self,
        source_tool: str,
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools suggested after executing source_tool.

        Args:
            source_tool: Name of tool that was just executed
            result: Result data from the tool execution
            context: Current session context

        Returns:
            List of suggested tools with priority and reasoning
        """
        if context is None:
            context = {}

        suggestions = []

        for relationship in TOOL_RELATIONSHIPS:
            if relationship.source_tool == source_tool:
                # Check if condition is met
                try:
                    if relationship.condition(result):
                        # Map context parameters
                        mapped_params = {}
                        for source_key, target_key in relationship.context_mapping.items():
                            if source_key in context:
                                mapped_params[target_key] = context[source_key]
                            elif source_key in result:
                                mapped_params[target_key] = result[source_key]

                        suggestions.append({
                            'tool': relationship.related_tool,
                            'priority': relationship.priority,
                            'reason': relationship.reason,
                            'params': mapped_params,
                            'relationship': relationship.relationship_type.value
                        })
                except Exception:
                    # Skip if condition evaluation fails
                    continue

        # Sort by priority (highest first)
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)


# Singleton instance
relationship_engine = ToolRelationshipEngine()
```

### Step 4: Integrate with Existing Tools

**Modify**: `tools/notebook.py`

Add at the top:
```python
from helpers.orchestration.notebook_orchestrator import notebook_orchestrator
```

Add new tool:
```python
@mcp.tool()
async def create_notebook_validated(
    workspace: str,
    notebook_name: str,
    template_type: str = "basic",
    validate: bool = True,
    optimize: bool = False
) -> str:
    """
    Create a PySpark notebook with automatic validation and optimization.

    This is an orchestrated operation that:
    1. Creates notebook from template
    2. Validates code (if validate=True)
    3. Analyzes performance (if optimize=True)
    4. Returns results with intelligent next-step suggestions

    Args:
        workspace: Fabric workspace name
        notebook_name: Name for the new notebook
        template_type: Template type (basic/etl/analytics/ml/fabric_integration/streaming)
        validate: Run validation after creation (default: True)
        optimize: Run performance analysis (default: False)

    Returns:
        JSON string with creation results, validation, and suggestions
    """
    ctx = get_context()
    workspace = workspace or ctx.workspace

    if not workspace:
        return json.dumps({
            "error": "No workspace specified and no workspace in context",
            "suggestion": "Use set_workspace() or provide workspace parameter"
        })

    result = await notebook_orchestrator.create_validated_notebook(
        workspace=workspace,
        notebook_name=notebook_name,
        template_type=template_type,
        validate=validate,
        optimize=optimize
    )

    return json.dumps(result, indent=2)
```

### Step 5: Test the Implementation

**Create**: `test_orchestration.py`

```python
"""Test orchestration functionality."""

import asyncio
from helpers.orchestration.notebook_orchestrator import notebook_orchestrator
from helpers.orchestration.tool_relationships import relationship_engine


async def test_create_validated_notebook():
    """Test notebook creation with validation."""
    print("Testing notebook orchestration...")

    result = await notebook_orchestrator.create_validated_notebook(
        workspace="TestWorkspace",
        notebook_name="TestNotebook",
        template_type="etl",
        validate=True,
        optimize=False
    )

    print("\n=== Orchestration Result ===")
    print(f"Success: {result.get('success')}")
    print(f"Steps completed: {result.get('steps_completed')}")

    if 'suggested_next_actions' in result:
        print("\n=== Suggested Next Actions ===")
        for suggestion in result['suggested_next_actions']:
            print(f"[Priority {suggestion['priority']}] {suggestion['tool']}")
            print(f"  Reason: {suggestion['reason']}")


def test_relationship_engine():
    """Test tool relationship suggestions."""
    print("\nTesting relationship engine...")

    # Simulate tool execution result
    result = {
        'success': True,
        'notebook_id': 'test-123',
        'created': True
    }

    context = {
        'workspace': 'TestWorkspace',
        'notebook_id': 'test-123'
    }

    suggestions = relationship_engine.get_suggested_tools(
        source_tool='create_pyspark_notebook',
        result=result,
        context=context
    )

    print("\n=== Tool Suggestions ===")
    for suggestion in suggestions:
        print(f"[Priority {suggestion['priority']}] {suggestion['tool']}")
        print(f"  Reason: {suggestion['reason']}")
        print(f"  Relationship: {suggestion['relationship']}")


if __name__ == "__main__":
    # Test orchestration
    asyncio.run(test_create_validated_notebook())

    # Test relationships
    test_relationship_engine()
```

Run test:
```bash
python test_orchestration.py
```

---

## Phase 2: Enhancement (Week 3-4)

### Step 1: Add More Tool Relationships

Expand `TOOL_RELATIONSHIPS` in `tool_relationships.py` to 30+ relationships covering:
- All notebook operations
- Lakehouse workflows
- Table exploration patterns
- Performance optimization chains

### Step 2: Create Workflow Chains

Add to `tool_relationships.py`:

```python
@dataclass
class WorkflowChain:
    """Pre-defined multi-step workflow."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    trigger_keywords: List[str]
    estimated_time: str

WORKFLOW_CHAINS = [
    WorkflowChain(
        name="complete_notebook_development",
        description="Full notebook development with validation and optimization",
        steps=[
            {'tool': 'create_pyspark_notebook', 'description': 'Create from template'},
            {'tool': 'validate_pyspark_code', 'description': 'Validate syntax'},
            {'tool': 'analyze_notebook_performance', 'description': 'Check performance'},
        ],
        trigger_keywords=["complete notebook", "production ready notebook"],
        estimated_time="30-60 seconds"
    ),
    # Add 2-3 more workflows
]
```

### Step 3: Implement Intent-Driven Execution

Add to `tools/notebook.py`:

```python
@mcp.tool()
async def execute_notebook_intent(
    goal: str,
    workspace: str | None = None,
    notebook_id: str | None = None
) -> str:
    """
    Execute notebook operations based on natural language intent.

    Supports intents like:
    - "Create a new ETL notebook with validation"
    - "Optimize my slow notebook"
    - "Explore all notebooks in workspace"

    Args:
        goal: What you want to achieve (natural language)
        workspace: Fabric workspace (optional, uses context)
        notebook_id: Notebook ID for existing notebooks (optional)

    Returns:
        JSON with results and suggestions
    """
    # Implementation from proposal document
    # ...
```

---

## Success Checklist

### Phase 1 Complete When:
- [ ] NotebookOrchestrator with 1 workflow implemented
- [ ] ToolRelationshipEngine with 10 relationships defined
- [ ] `create_notebook_validated` tool working
- [ ] Tests passing
- [ ] Documentation updated

### Phase 2 Complete When:
- [ ] 30+ tool relationships defined
- [ ] 3 workflow chains implemented
- [ ] Intent-driven execution working
- [ ] All tools return suggestions
- [ ] Comprehensive testing done

---

## Next Steps

1. **Review this implementation guide**
2. **Start with Phase 1, Step 1** (directory structure)
3. **Implement one component at a time**
4. **Test each component before moving forward**
5. **Iterate based on real usage patterns**

---

## Support

Reference documents:
- [AGENTIC_ARCHITECTURE_PROPOSAL.md](./AGENTIC_ARCHITECTURE_PROPOSAL.md) - Full architectural design
- [COMPARISON_POWERBI_VS_FABRIC_MCP.md](./COMPARISON_POWERBI_VS_FABRIC_MCP.md) - Pattern analysis

Questions? Issues? Create a GitHub issue or discussion.
