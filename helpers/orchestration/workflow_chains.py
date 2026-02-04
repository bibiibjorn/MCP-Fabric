"""Pre-defined workflow chains for common multi-step operations."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class WorkflowStep:
    """Individual step in a workflow chain."""
    tool: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    conditional: bool = False
    condition: Optional[str] = None
    retry_on_failure: bool = False
    required: bool = True


@dataclass
class WorkflowChain:
    """Pre-defined multi-step workflow."""
    name: str
    description: str
    steps: List[WorkflowStep]
    trigger_keywords: List[str]
    estimated_time: str
    category: str = "general"
    requires_context: List[str] = field(default_factory=list)


# Define comprehensive workflow chains
WORKFLOW_CHAINS = [
    WorkflowChain(
        name="complete_notebook_development",
        description="Full notebook development lifecycle with validation and optimization",
        category="notebook",
        trigger_keywords=[
            "complete notebook", "production ready notebook", "full notebook development",
            "create and validate notebook", "production notebook"
        ],
        estimated_time="30-60 seconds",
        requires_context=["workspace"],
        steps=[
            WorkflowStep(
                tool="create_pyspark_notebook",
                description="Create notebook from template",
                params={"template_type": "etl"}
            ),
            WorkflowStep(
                tool="get_notebook_content",
                description="Retrieve notebook content",
                params={}
            ),
            WorkflowStep(
                tool="validate_pyspark_code",
                description="Validate PySpark syntax and best practices",
                params={}
            ),
            WorkflowStep(
                tool="validate_fabric_code",
                description="Validate Fabric API compatibility",
                params={},
                conditional=True,
                condition="validation_passed"
            ),
            WorkflowStep(
                tool="analyze_notebook_performance",
                description="Analyze performance and optimization opportunities",
                params={},
                conditional=True,
                condition="validation_passed"
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate data quality checks",
                params={"operation": "data_quality"},
                conditional=True,
                condition="template_is_etl",
                required=False
            ),
        ]
    ),

    WorkflowChain(
        name="lakehouse_data_exploration",
        description="Comprehensive lakehouse and table discovery with schema analysis",
        category="lakehouse",
        trigger_keywords=[
            "explore lakehouse", "discover data", "data exploration",
            "understand data structure", "lakehouse discovery", "explore tables"
        ],
        estimated_time="15-30 seconds",
        requires_context=["workspace"],
        steps=[
            WorkflowStep(
                tool="list_lakehouses",
                description="List all lakehouses in workspace",
                params={}
            ),
            WorkflowStep(
                tool="list_tables",
                description="List tables in each lakehouse",
                params={}
            ),
            WorkflowStep(
                tool="get_all_lakehouse_schemas",
                description="Get detailed schemas for all tables",
                params={}
            ),
            WorkflowStep(
                tool="run_lakehouse_query",
                description="Sample data from key tables",
                params={"query": "SELECT * FROM {table} LIMIT 10"},
                conditional=True,
                condition="tables_found",
                required=False
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate sample reading code",
                params={"operation": "read_table"},
                conditional=True,
                condition="tables_found",
                required=False
            ),
        ]
    ),

    WorkflowChain(
        name="notebook_optimization_pipeline",
        description="Analyze and optimize existing notebook performance",
        category="optimization",
        trigger_keywords=[
            "optimize notebook", "improve performance", "notebook slow",
            "speed up notebook", "performance tuning", "optimize code"
        ],
        estimated_time="20-40 seconds",
        requires_context=["workspace", "notebook_id"],
        steps=[
            WorkflowStep(
                tool="get_notebook_content",
                description="Retrieve current notebook content",
                params={}
            ),
            WorkflowStep(
                tool="validate_pyspark_code",
                description="Validate code for errors",
                params={}
            ),
            WorkflowStep(
                tool="analyze_notebook_performance",
                description="Identify performance bottlenecks",
                params={}
            ),
            WorkflowStep(
                tool="get_notebook_optimization_suggestions",
                description="Get specific optimization recommendations",
                params={},
                conditional=True,
                condition="performance_score_low"
            ),
            WorkflowStep(
                tool="generate_fabric_code",
                description="Generate optimized code",
                params={"operation": "performance_optimization"},
                conditional=True,
                condition="performance_score_low"
            ),
        ]
    ),

    WorkflowChain(
        name="etl_pipeline_setup",
        description="Complete ETL pipeline setup from lakehouse to notebook",
        category="etl",
        trigger_keywords=[
            "setup etl", "create etl pipeline", "etl workflow",
            "data pipeline", "build etl", "etl notebook"
        ],
        estimated_time="45-90 seconds",
        requires_context=["workspace"],
        steps=[
            WorkflowStep(
                tool="list_lakehouses",
                description="Identify source lakehouses",
                params={}
            ),
            WorkflowStep(
                tool="list_tables",
                description="Discover source tables",
                params={}
            ),
            WorkflowStep(
                tool="get_all_lakehouse_schemas",
                description="Analyze table schemas",
                params={}
            ),
            WorkflowStep(
                tool="create_pyspark_notebook",
                description="Create ETL notebook",
                params={"template_type": "etl"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate read operations",
                params={"operation": "read_table"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate transformation logic",
                params={"operation": "transform"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate data quality checks",
                params={"operation": "data_quality"}
            ),
            WorkflowStep(
                tool="validate_pyspark_code",
                description="Validate complete ETL code",
                params={},
                retry_on_failure=True
            ),
        ]
    ),

    WorkflowChain(
        name="ml_notebook_setup",
        description="Machine learning notebook setup with best practices",
        category="ml",
        trigger_keywords=[
            "ml notebook", "machine learning", "create ml pipeline",
            "ml workflow", "model training", "ml setup"
        ],
        estimated_time="30-60 seconds",
        requires_context=["workspace"],
        steps=[
            WorkflowStep(
                tool="list_lakehouses",
                description="Identify data sources",
                params={}
            ),
            WorkflowStep(
                tool="create_pyspark_notebook",
                description="Create ML notebook from template",
                params={"template_type": "ml"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate feature engineering code",
                params={"operation": "ml_features"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate model training code",
                params={"operation": "ml_training"}
            ),
            WorkflowStep(
                tool="validate_pyspark_code",
                description="Validate ML code",
                params={}
            ),
            WorkflowStep(
                tool="get_notebook_best_practices",
                description="Get ML best practices",
                params={"category": "ml"}
            ),
        ]
    ),

    WorkflowChain(
        name="notebook_validation_complete",
        description="Comprehensive notebook validation and quality checks",
        category="validation",
        trigger_keywords=[
            "validate notebook completely", "full validation", "check notebook quality",
            "comprehensive validation", "validate everything"
        ],
        estimated_time="15-30 seconds",
        requires_context=["workspace", "notebook_id"],
        steps=[
            WorkflowStep(
                tool="get_notebook_content",
                description="Retrieve notebook content",
                params={}
            ),
            WorkflowStep(
                tool="validate_pyspark_code",
                description="Validate PySpark syntax",
                params={}
            ),
            WorkflowStep(
                tool="validate_fabric_code",
                description="Validate Fabric compatibility",
                params={}
            ),
            WorkflowStep(
                tool="get_notebook_best_practices",
                description="Check against best practices",
                params={}
            ),
            WorkflowStep(
                tool="analyze_notebook_performance",
                description="Analyze performance characteristics",
                params={}
            ),
        ]
    ),

    WorkflowChain(
        name="data_quality_framework",
        description="Setup comprehensive data quality framework",
        category="data_quality",
        trigger_keywords=[
            "data quality", "quality checks", "data validation",
            "quality framework", "data integrity"
        ],
        estimated_time="20-40 seconds",
        requires_context=["workspace", "lakehouse"],
        steps=[
            WorkflowStep(
                tool="list_tables",
                description="Identify tables for quality checks",
                params={}
            ),
            WorkflowStep(
                tool="get_all_lakehouse_schemas",
                description="Analyze table schemas",
                params={}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate null checks",
                params={"operation": "data_quality", "check_type": "null_checks"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate duplicate detection",
                params={"operation": "data_quality", "check_type": "duplicates"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate data type validation",
                params={"operation": "data_quality", "check_type": "type_validation"}
            ),
            WorkflowStep(
                tool="create_pyspark_notebook",
                description="Create data quality notebook",
                params={"template_type": "analytics"}
            ),
        ]
    ),

    WorkflowChain(
        name="streaming_pipeline_setup",
        description="Setup real-time streaming data pipeline",
        category="streaming",
        trigger_keywords=[
            "streaming", "real-time", "stream processing",
            "streaming pipeline", "real-time data"
        ],
        estimated_time="30-50 seconds",
        requires_context=["workspace"],
        steps=[
            WorkflowStep(
                tool="create_pyspark_notebook",
                description="Create streaming notebook",
                params={"template_type": "streaming"}
            ),
            WorkflowStep(
                tool="generate_fabric_code",
                description="Generate streaming source configuration",
                params={"operation": "streaming_source"}
            ),
            WorkflowStep(
                tool="generate_pyspark_code",
                description="Generate stream processing logic",
                params={"operation": "streaming_transform"}
            ),
            WorkflowStep(
                tool="generate_fabric_code",
                description="Generate streaming sink configuration",
                params={"operation": "streaming_sink"}
            ),
            WorkflowStep(
                tool="validate_fabric_code",
                description="Validate streaming configuration",
                params={}
            ),
        ]
    ),
]


class WorkflowEngine:
    """Engine for managing and executing workflow chains."""

    def __init__(self):
        self.workflows = WORKFLOW_CHAINS

    def find_workflow(self, query: str) -> Optional[WorkflowChain]:
        """
        Find a workflow matching the query string.

        Args:
            query: User query or intent

        Returns:
            Matching WorkflowChain or None
        """
        query_lower = query.lower()

        # Try exact keyword matching first
        for workflow in self.workflows:
            if any(keyword in query_lower for keyword in workflow.trigger_keywords):
                return workflow

        return None

    def get_workflows_by_category(self, category: str) -> List[WorkflowChain]:
        """Get all workflows in a specific category."""
        return [w for w in self.workflows if w.category == category]

    def list_all_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows with metadata."""
        return [
            {
                'name': w.name,
                'description': w.description,
                'category': w.category,
                'estimated_time': w.estimated_time,
                'steps_count': len(w.steps),
                'trigger_keywords': w.trigger_keywords[:3]  # First 3 keywords
            }
            for w in self.workflows
        ]

    def validate_workflow_context(
        self,
        workflow: WorkflowChain,
        context: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate if required context is available for workflow.

        Args:
            workflow: WorkflowChain to validate
            context: Current session context

        Returns:
            Tuple of (is_valid, missing_items)
        """
        missing = []
        for required_key in workflow.requires_context:
            if required_key not in context or not context[required_key]:
                missing.append(required_key)

        return len(missing) == 0, missing

    def add_workflow(self, workflow: WorkflowChain):
        """Add a new workflow dynamically."""
        self.workflows.append(workflow)


# Singleton instance
workflow_engine = WorkflowEngine()
