"""Tool relationship definitions for intelligent suggestions."""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any
from enum import Enum


class RelationType(Enum):
    """Types of relationships between tools."""
    REQUIRES = "requires"     # Must run before
    SUGGESTS = "suggests"     # Should consider running
    ENRICHES = "enriches"     # Adds value to
    VALIDATES = "validates"   # Checks result of
    INVERSE = "inverse"       # Opposite operation


@dataclass
class ToolRelationship:
    """Defines relationship between two tools."""
    source_tool: str
    related_tool: str
    relationship_type: RelationType
    condition: Callable[[Dict], bool]
    priority: int  # 1-10, higher = more important
    reason: str
    context_mapping: Dict[str, str] = field(default_factory=dict)
    example_params: Dict[str, Any] = field(default_factory=dict)


# Define comprehensive tool relationships (50+)
TOOL_RELATIONSHIPS = [
    # ===== Notebook Creation & Validation Workflows =====
    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('success', False) or 'notebook_id' in result,
        priority=9,
        reason="Validate new notebook for syntax and best practices",
        context_mapping={'notebook_id': 'code_source'}
    ),

    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="get_notebook_content",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: 'notebook_id' in result,
        priority=6,
        reason="View notebook content after creation",
        context_mapping={'notebook_id': 'notebook_id', 'workspace': 'workspace'}
    ),

    ToolRelationship(
        source_tool="validate_pyspark_code",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('has_issues', False) or result.get('has_errors', False),
        priority=8,
        reason="Generate corrected code for identified issues"
    ),

    ToolRelationship(
        source_tool="validate_pyspark_code",
        related_tool="analyze_notebook_performance",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: not result.get('has_errors', False),
        priority=7,
        reason="Analyze performance after validation passes",
        context_mapping={'notebook_id': 'notebook_id'}
    ),

    ToolRelationship(
        source_tool="analyze_notebook_performance",
        related_tool="generate_fabric_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('performance_score', 100) < 70 or result.get('score', 100) < 70,
        priority=9,
        reason="Low performance score - generate optimized code",
        example_params={'operation': 'performance_optimization'}
    ),

    # ===== Lakehouse & Table Discovery Workflows =====
    ToolRelationship(
        source_tool="list_workspaces",
        related_tool="list_lakehouses",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('workspaces', [])) > 0,
        priority=7,
        reason="Explore lakehouses in discovered workspaces",
        context_mapping={'first_workspace': 'workspace'}
    ),

    ToolRelationship(
        source_tool="list_lakehouses",
        related_tool="list_tables",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('lakehouses', [])) > 0,
        priority=8,
        reason="Explore tables in discovered lakehouses",
        context_mapping={'first_lakehouse': 'lakehouse'}
    ),

    ToolRelationship(
        source_tool="list_tables",
        related_tool="get_all_lakehouse_schemas",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('tables', [])) > 0,
        priority=7,
        reason="Get detailed schemas for all tables found",
        context_mapping={'lakehouse': 'lakehouse', 'workspace': 'workspace'}
    ),

    ToolRelationship(
        source_tool="get_lakehouse_table_schema",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'columns' in result or 'schema' in result,
        priority=8,
        reason="Generate PySpark code to work with this table",
        example_params={'operation': 'read_table'}
    ),

    ToolRelationship(
        source_tool="get_all_lakehouse_schemas",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('tables', [])) > 0,
        priority=7,
        reason="Generate code to read from discovered tables",
        example_params={'operation': 'read_table'}
    ),

    # ===== Lakehouse Creation & Integration =====
    ToolRelationship(
        source_tool="create_lakehouse",
        related_tool="create_pyspark_notebook",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('created', False),
        priority=8,
        reason="Create ETL notebook for new lakehouse",
        example_params={'template_type': 'etl'},
        context_mapping={'lakehouse': 'lakehouse_name', 'workspace': 'workspace'}
    ),

    ToolRelationship(
        source_tool="create_lakehouse",
        related_tool="list_tables",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('created', False),
        priority=6,
        reason="Verify lakehouse structure and tables",
        context_mapping={'lakehouse': 'lakehouse', 'workspace': 'workspace'}
    ),

    # ===== Code Generation & Validation =====
    ToolRelationship(
        source_tool="generate_pyspark_code",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.VALIDATES,
        condition=lambda result: 'code' in result,
        priority=7,
        reason="Validate generated code before use"
    ),

    ToolRelationship(
        source_tool="generate_fabric_code",
        related_tool="validate_fabric_code",
        relationship_type=RelationType.VALIDATES,
        condition=lambda result: 'code' in result,
        priority=7,
        reason="Validate Fabric-specific code for compatibility"
    ),

    # ===== Template & Best Practices =====
    ToolRelationship(
        source_tool="list_notebook_templates",
        related_tool="create_pyspark_notebook",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('templates', [])) > 0,
        priority=6,
        reason="Create notebook using discovered templates"
    ),

    ToolRelationship(
        source_tool="get_notebook_best_practices",
        related_tool="validate_pyspark_code",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: True,
        priority=5,
        reason="Apply best practices during validation"
    ),

    # ===== Notebook Management =====
    ToolRelationship(
        source_tool="list_notebooks",
        related_tool="get_notebook_content",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('notebooks', [])) > 0,
        priority=5,
        reason="View content of discovered notebooks",
        context_mapping={'first_notebook': 'notebook_id'}
    ),

    ToolRelationship(
        source_tool="get_notebook_content",
        related_tool="analyze_notebook_performance",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'content' in result or 'code' in result,
        priority=6,
        reason="Analyze performance of notebook code",
        context_mapping={'notebook_id': 'notebook_id'}
    ),

    # ===== Query Execution & Analysis =====
    ToolRelationship(
        source_tool="run_lakehouse_query",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'results' in result,
        priority=6,
        reason="Generate PySpark code based on query results",
        example_params={'operation': 'transform'}
    ),

    # ===== Public API Documentation =====
    ToolRelationship(
        source_tool="search_public_apis",
        related_tool="get_public_api_details",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('results', [])) > 0,
        priority=7,
        reason="Get detailed documentation for discovered APIs"
    ),

    ToolRelationship(
        source_tool="get_public_api_details",
        related_tool="generate_fabric_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'api_info' in result,
        priority=6,
        reason="Generate code using API documentation",
        example_params={'operation': 'api_call'}
    ),

    # ===== Advanced Notebook Workflows =====
    ToolRelationship(
        source_tool="create_pyspark_notebook",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('template_type', '') in ['basic', 'empty'],
        priority=7,
        reason="Add functionality to basic notebook template",
        example_params={'operation': 'etl'}
    ),

    ToolRelationship(
        source_tool="analyze_notebook_performance",
        related_tool="get_notebook_optimization_suggestions",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: len(result.get('issues', [])) > 0,
        priority=8,
        reason="Get specific optimization recommendations",
        context_mapping={'notebook_id': 'notebook_id'}
    ),

    # ===== Data Quality & Validation =====
    ToolRelationship(
        source_tool="get_lakehouse_table_schema",
        related_tool="generate_pyspark_code",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: 'columns' in result,
        priority=7,
        reason="Generate data quality checks for table columns",
        example_params={'operation': 'data_quality'}
    ),

    # ===== Workflow Completion Chains =====
    ToolRelationship(
        source_tool="validate_fabric_code",
        related_tool="create_pyspark_notebook",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: not result.get('has_errors', False) and result.get('validated', False),
        priority=6,
        reason="Validated code ready to be added to notebook"
    ),

    # ===== Error Recovery & Debugging =====
    ToolRelationship(
        source_tool="validate_pyspark_code",
        related_tool="get_validation_help",
        relationship_type=RelationType.SUGGESTS,
        condition=lambda result: result.get('has_errors', False),
        priority=9,
        reason="Get help resolving validation errors"
    ),

    # ===== Multi-Lakehouse Operations =====
    ToolRelationship(
        source_tool="list_lakehouses",
        related_tool="get_all_lakehouse_schemas",
        relationship_type=RelationType.ENRICHES,
        condition=lambda result: len(result.get('lakehouses', [])) > 1,
        priority=6,
        reason="Compare schemas across multiple lakehouses",
        context_mapping={'lakehouse': 'lakehouse'}
    ),
]


class ToolRelationshipEngine:
    """Engine for discovering and suggesting related tools."""

    def __init__(self):
        self.relationships = TOOL_RELATIONSHIPS

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

        for relationship in self.relationships:
            if relationship.source_tool == source_tool:
                # Check if condition is met
                try:
                    if relationship.condition(result):
                        # Map context parameters
                        mapped_params = {}
                        for source_key, target_key in relationship.context_mapping.items():
                            # Try to find value in context first, then in result
                            if source_key in context:
                                mapped_params[target_key] = context[source_key]
                            elif source_key in result:
                                mapped_params[target_key] = result[source_key]
                            # Handle special cases like 'first_workspace', 'first_lakehouse'
                            elif source_key == 'first_workspace' and 'workspaces' in result:
                                workspaces = result.get('workspaces', [])
                                if workspaces:
                                    mapped_params[target_key] = workspaces[0] if isinstance(workspaces[0], str) else workspaces[0].get('name')
                            elif source_key == 'first_lakehouse' and 'lakehouses' in result:
                                lakehouses = result.get('lakehouses', [])
                                if lakehouses:
                                    mapped_params[target_key] = lakehouses[0] if isinstance(lakehouses[0], str) else lakehouses[0].get('name')
                            elif source_key == 'first_notebook' and 'notebooks' in result:
                                notebooks = result.get('notebooks', [])
                                if notebooks:
                                    mapped_params[target_key] = notebooks[0] if isinstance(notebooks[0], str) else notebooks[0].get('id')

                        # Add example params if available
                        if relationship.example_params:
                            mapped_params.update(relationship.example_params)

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

    def add_relationship(self, relationship: ToolRelationship):
        """Add a new tool relationship dynamically."""
        self.relationships.append(relationship)

    def get_relationships_for_tool(self, tool_name: str) -> List[ToolRelationship]:
        """Get all relationships where the tool is the source."""
        return [r for r in self.relationships if r.source_tool == tool_name]

    def get_inverse_relationships(self, tool_name: str) -> List[ToolRelationship]:
        """Get all relationships where the tool is the target."""
        return [r for r in self.relationships if r.related_tool == tool_name]


# Singleton instance
relationship_engine = ToolRelationshipEngine()
