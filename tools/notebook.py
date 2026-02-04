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
async def list_notebooks(workspace: Optional[str] = None, ctx: Context = None) -> str:
    """List all notebooks in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace (optional)
        ctx: Context object containing client information
    Returns:
        A string containing the list of notebooks or an error message.
    """

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


@mcp.tool()
async def create_notebook(
    workspace: str,
    # notebook_name: str,
    # content: str,
    ctx: Context = None,
) -> str:
    """Create a new notebook in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace
        notebook_name: Name of the new notebook
        content: Content of the notebook (in JSON format)
        ctx: Context object containing client information
    Returns:
        A string containing the ID of the created notebook or an error message.
    """
    notebook_json = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "source": ["print('Hello, Fabric!')\n"],
                "execution_count": None,
                "outputs": [],
                "metadata": {},
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
    }
    notebook_content = json.dumps(notebook_json)
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        notebook_client = NotebookClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )
        response = await notebook_client.create_notebook(
            workspace, "test_notebook_2", notebook_content
        )
        return response.get("id", "")  # Return the notebook ID or an empty string
    except Exception as e:
        logger.error(f"Error creating notebook: {str(e)}")
        return f"Error creating notebook: {str(e)}"


@mcp.tool()
async def get_notebook_content(
    workspace: str,
    notebook_id: str,
    ctx: Context = None
) -> str:
    """Get the content of a specific notebook in a Fabric workspace.

    This retrieves the actual notebook script/code by calling the getDefinition
    endpoint, which returns the full notebook content including all cells.

    Args:
        workspace: Name or ID of the workspace
        notebook_id: ID or name of the notebook
        ctx: Context object containing client information
    Returns:
        A string containing the notebook content in JSON format or an error message.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

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


@mcp.tool()
async def create_pyspark_notebook(
    workspace: str,
    notebook_name: str,
    template_type: str = "basic",
    ctx: Context = None,
) -> str:
    """Create a new PySpark notebook from a template in a Fabric workspace.

    Args:
        workspace: Name or ID of the workspace
        notebook_name: Name of the new notebook
        template_type: Type of PySpark template ('basic', 'etl', 'analytics', 'ml')
        ctx: Context object containing client information
    Returns:
        A string containing the ID of the created notebook or an error message.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        # Define PySpark templates
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
            return f"Invalid template type. Available templates: {', '.join(templates.keys())}"

        # Create notebook JSON structure
        notebook_json = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": templates[template_type]["cells"],
            "metadata": {
                "language_info": {"name": "python"},
                "kernel_info": {"name": "synapse_pyspark"},
                "description": f"PySpark notebook created from {template_type} template"
            },
        }
        
        notebook_content = json.dumps(notebook_json, indent=2)

        notebook_client = NotebookClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )
        response = await notebook_client.create_notebook(
            workspace, notebook_name, notebook_content
        )
        
        if isinstance(response, dict) and response.get("id"):
            return f"Created PySpark notebook '{notebook_name}' with ID: {response['id']}"
        else:
            return f"Failed to create notebook: {response}"
            
    except Exception as e:
        logger.error(f"Error creating PySpark notebook: {str(e)}")
        return f"Error creating PySpark notebook: {str(e)}"

@mcp.tool()
async def generate_pyspark_code(
    operation: str,
    source_table: Optional[str] = None,
    target_table: Optional[str] = None,
    columns: Optional[str] = None,
    filter_condition: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Generate PySpark code for common operations.

    Args:
        operation: Type of operation ('read_table', 'write_table', 'transform', 'join', 'aggregate')
        source_table: Source table name (format: lakehouse.table_name)
        target_table: Target table name (format: lakehouse.table_name)
        columns: Comma-separated list of columns
        filter_condition: Filter condition for data
        ctx: Context object containing client information
    Returns:
        A string containing the generated PySpark code or an error message.
    """
    try:
        code_templates = {
            "read_table": f"""# Read data from table
df = spark.table("{source_table or 'lakehouse.table_name'}")
df.show()
df.printSchema()""",
            
            "write_table": f"""# Write data to table
df.write \\
    .format("delta") \\
    .mode("overwrite") \\
    .saveAsTable("{target_table or 'lakehouse.output_table'}")

print(f"Successfully wrote {{df.count()}} records to {target_table or 'lakehouse.output_table'}")""",
            
            "transform": f"""# Data transformation
from pyspark.sql.functions import *

df_transformed = df \\
    .select({columns or '*'}) \\
    {f'.filter({filter_condition})' if filter_condition else ''} \\
    .withColumn("processed_date", current_timestamp())

df_transformed.show()""",
            
            "join": f"""# Join tables
df1 = spark.table("{source_table or 'lakehouse.table1'}")
df2 = spark.table("{target_table or 'lakehouse.table2'}")

# Inner join (modify join condition as needed)
df_joined = df1.join(df2, df1.id == df2.id, "inner")

df_joined.show()""",
            
            "aggregate": f"""# Data aggregation
from pyspark.sql.functions import *

df_agg = df \\
    .groupBy({columns or '"column1"'}) \\
    .agg(
        count("*").alias("count"),
        sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
        max("date").alias("max_date")
    ) \\
    .orderBy(desc("total_amount"))

df_agg.show()""",
            
            "schema_inference": f"""# Schema inference and data profiling
print("=== Schema Information ===")
df.printSchema()

print("\\n=== Data Profile ===")
print(f"Record count: {{df.count()}}")
print(f"Column count: {{len(df.columns)}}")

print("\\n=== Column Statistics ===")
df.describe().show()

print("\\n=== Null Value Analysis ===")
from pyspark.sql.functions import col, sum as spark_sum, isnan, when, count

null_counts = df.select([
    spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c)
    for c in df.columns
])
null_counts.show()""",
            
            "data_quality": f"""# Data quality checks
from pyspark.sql.functions import *

print("=== Data Quality Report ===")

# Check for duplicates
duplicate_count = df.count() - df.distinct().count()
print(f"Duplicate rows: {{duplicate_count}}")

# Check for null values
total_rows = df.count()
for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    null_percentage = (null_count / total_rows) * 100
    print(f"{{column}}: {{null_count}} nulls ({{null_percentage:.2f}}%)")

# Check data ranges (for numeric columns)
numeric_columns = [field.name for field in df.schema.fields 
                  if field.dataType.simpleString() in ['int', 'double', 'float', 'bigint']]

if numeric_columns:
    print("\\n=== Numeric Column Ranges ===")
    df.select([
        min(col(c)).alias(f"{c}_min"),
        max(col(c)).alias(f"{c}_max")
        for c in numeric_columns
    ]).show()""",
            
            "performance_optimization": f"""# Performance optimization techniques

# 1. Cache frequently used DataFrames
df.cache()
print(f"Cached DataFrame with {{df.count()}} records")

# 2. Repartition for better parallelism
optimal_partitions = spark.sparkContext.defaultParallelism * 2
df_repartitioned = df.repartition(optimal_partitions)

# 3. Use broadcast for small dimension tables (< 200MB)
from pyspark.sql.functions import broadcast
# df_joined = large_df.join(broadcast(small_df), "key")

# 4. Optimize file formats - use Delta Lake
df.write \\
    .format("delta") \\
    .mode("overwrite") \\
    .option("optimizeWrite", "true") \\
    .option("autoOptimize", "true") \\
    .saveAsTable("{target_table or 'lakehouse.optimized_table'}")

# 5. Show execution plan
df.explain(True)"""
        }
        
        if operation not in code_templates:
            available_ops = ", ".join(code_templates.keys())
            return f"Invalid operation. Available operations: {available_ops}"
        
        generated_code = code_templates[operation]
        
        return f"""```python
{generated_code}
```

**Generated PySpark code for '{operation}' operation**

This code can be copied into a notebook cell and executed. Remember to:
- Replace placeholder table names with actual table names
- Adjust column names and conditions as needed
- Test with a small dataset first
- Review the execution plan for performance optimization"""
        
    except Exception as e:
        logger.error(f"Error generating PySpark code: {str(e)}")
        return f"Error generating PySpark code: {str(e)}"

@mcp.tool()
async def validate_pyspark_code(
    code: str,
    ctx: Context = None,
) -> str:
    """Validate PySpark code for syntax and best practices.

    Args:
        code: PySpark code to validate
        ctx: Context object containing client information
    Returns:
        A string containing validation results and suggestions.
    """
    try:
        validation_results = []
        warnings = []
        suggestions = []
        
        # Basic syntax validation
        try:
            compile(code, '<string>', 'exec')
            validation_results.append("✅ Syntax validation: PASSED")
        except SyntaxError as e:
            validation_results.append(f"❌ Syntax validation: FAILED - {e}")
            return "\n".join(validation_results)
        
        # PySpark best practices checks
        lines = code.split('\n')
        
        # Check for common imports
        has_spark_imports = any('from pyspark' in line or 'import pyspark' in line for line in lines)
        if not has_spark_imports:
            warnings.append("⚠️  No PySpark imports detected. Add: from pyspark.sql import SparkSession")
        
        # Check for DataFrame operations
        has_df_operations = any('df.' in line or '.show()' in line for line in lines)
        if has_df_operations:
            validation_results.append("✅ DataFrame operations detected")
        
        # Check for performance anti-patterns
        if '.collect()' in code:
            warnings.append("⚠️  .collect() detected - avoid on large datasets, use .show() or .take() instead")
        
        if '.toPandas()' in code:
            warnings.append("⚠️  .toPandas() detected - ensure dataset fits in driver memory")
        
        if 'for row in df.collect()' in code:
            warnings.append("❌ Anti-pattern: iterating over collected DataFrame. Use DataFrame operations instead")
        
        # Check for caching opportunities
        df_count = code.count('df.')
        if df_count > 3 and '.cache()' not in code and '.persist()' not in code:
            suggestions.append("💡 Consider caching DataFrame with .cache() for repeated operations")
        
        # Check for schema definition
        if 'createDataFrame' in code and 'StructType' not in code:
            suggestions.append("💡 Consider defining explicit schema when creating DataFrames")
        
        # Check for null handling
        if '.filter(' in code and 'isNull' not in code and 'isNotNull' not in code:
            suggestions.append("💡 Consider adding null value handling in filters")
        
        # Check for partitioning
        if '.write.' in code and 'partitionBy' not in code:
            suggestions.append("💡 Consider partitioning data when writing large datasets")
        
        # Check for Delta Lake usage
        if '.write.' in code and 'format("delta")' not in code:
            suggestions.append("💡 Consider using Delta Lake format for ACID transactions and time travel")
        
        # Compile results
        result = "# PySpark Code Validation Report\n\n"
        result += "## Validation Results\n"
        result += "\n".join(validation_results) + "\n\n"
        
        if warnings:
            result += "## Warnings\n"
            result += "\n".join(warnings) + "\n\n"
        
        if suggestions:
            result += "## Optimization Suggestions\n"
            result += "\n".join(suggestions) + "\n\n"
        
        if not warnings and not suggestions:
            result += "## Summary\n✅ Code looks good! No issues detected.\n"
        else:
            result += f"## Summary\n📊 Found {len(warnings)} warnings and {len(suggestions)} optimization opportunities.\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating PySpark code: {str(e)}")
        return f"Error validating PySpark code: {str(e)}"

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


@mcp.tool()
async def update_notebook_cell(
    workspace: str,
    notebook_id: str,
    cell_index: int,
    cell_content: str,
    cell_type: str = "code",
    ctx: Context = None,
) -> str:
    """Update a specific cell in a notebook.

    Args:
        workspace: Name or ID of the workspace
        notebook_id: ID or name of the notebook
        cell_index: Index of the cell to update (0-based)
        cell_content: New content for the cell
        cell_type: Type of cell ('code' or 'markdown')
        ctx: Context object containing client information
    Returns:
        A string confirming the update or an error message.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        notebook_client = NotebookClient(fabric_client)

        # Resolve workspace ID
        workspace_id = await fabric_client.resolve_workspace(workspace)

        # Resolve notebook ID and get notebook name
        if not _is_valid_uuid(notebook_id):
            notebook_name = notebook_id
            notebook_resolved_id = await fabric_client.resolve_item_id(
                item=notebook_id, type="Notebook", workspace=workspace_id
            )
        else:
            notebook_resolved_id = notebook_id
            # Get notebook metadata to get the name
            notebook_meta = await notebook_client.get_notebook(workspace_id, notebook_resolved_id)
            if isinstance(notebook_meta, dict):
                notebook_name = notebook_meta.get("displayName", "notebook")
            else:
                notebook_name = "notebook"

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
            notebook_name=notebook_name,
        )

        # Check for errors in the result
        if result is None:
            return "Error: Update API returned None - the operation may have failed"

        if isinstance(result, dict) and "error" in result:
            return f"Error updating notebook: {result['error']}"

        return f"Successfully updated cell {cell_index} in notebook '{notebook_name}' with {cell_type} content ({len(cell_content)} characters)."

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing notebook JSON: {str(e)}")
        return f"Error parsing notebook content: {str(e)}"
    except Exception as e:
        logger.error(f"Error updating notebook cell: {str(e)}")
        return f"Error updating notebook cell: {str(e)}"

@mcp.tool()
async def create_fabric_notebook(
    workspace: str,
    notebook_name: str,
    template_type: str = "fabric_integration",
    ctx: Context = None,
) -> str:
    """Create a new notebook optimized for Microsoft Fabric using advanced templates.

    Args:
        workspace: Name or ID of the workspace
        notebook_name: Name of the new notebook
        template_type: Type of Fabric template ('fabric_integration', 'streaming')
        ctx: Context object containing client information
    Returns:
        A string containing the ID of the created notebook or an error message.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        from helpers.pyspark_helpers import create_notebook_from_template
        
        # Create notebook from advanced template
        notebook_data = create_notebook_from_template(template_type)
        notebook_content = json.dumps(notebook_data, indent=2)

        notebook_client = NotebookClient(
            FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        )
        response = await notebook_client.create_notebook(
            workspace, notebook_name, notebook_content
        )
        
        if isinstance(response, dict) and response.get("id"):
            return f"Created Fabric-optimized notebook '{notebook_name}' with ID: {response['id']} using {template_type} template"
        else:
            return f"Failed to create notebook: {response}"
            
    except Exception as e:
        logger.error(f"Error creating Fabric notebook: {str(e)}")
        return f"Error creating Fabric notebook: {str(e)}"

@mcp.tool()
async def generate_fabric_code(
    operation: str,
    lakehouse_name: Optional[str] = None,
    table_name: Optional[str] = None,
    target_table: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Generate Fabric-specific PySpark code for lakehouse operations.

    Args:
        operation: Type of operation ('read_lakehouse', 'write_lakehouse', 'merge_delta', 'performance_monitor')
        lakehouse_name: Name of the lakehouse
        table_name: Name of the source table
        target_table: Name of the target table (for write/merge operations)
        ctx: Context object containing client information
    Returns:
        A string containing the generated Fabric-specific PySpark code.
    """
    try:
        from helpers.pyspark_helpers import PySparkCodeGenerator
        
        generator = PySparkCodeGenerator()
        
        if operation == "read_lakehouse":
            if not lakehouse_name or not table_name:
                return "Error: lakehouse_name and table_name are required for read_lakehouse operation"
            code = generator.generate_fabric_lakehouse_reader(lakehouse_name, table_name)
            
        elif operation == "write_lakehouse":
            if not table_name:
                return "Error: table_name is required for write_lakehouse operation"
            code = generator.generate_fabric_lakehouse_writer(table_name)
            
        elif operation == "merge_delta":
            if not target_table:
                return "Error: target_table is required for merge_delta operation"
            source_df = "new_df"  # Default source DataFrame name
            join_condition = "target.id = source.id"  # Default join condition
            code = generator.generate_delta_merge_operation(target_table, source_df, join_condition)
            
        elif operation == "performance_monitor":
            code = generator.generate_performance_monitoring()
            
        else:
            available_ops = ["read_lakehouse", "write_lakehouse", "merge_delta", "performance_monitor"]
            return f"Invalid operation. Available operations: {', '.join(available_ops)}"
        
        return f"""```python
{code}
```

**Generated Fabric-specific PySpark code for '{operation}' operation**

This code is optimized for Microsoft Fabric and includes:
- Proper Delta Lake integration
- Fabric lakehouse connectivity
- Performance monitoring capabilities
- Best practices for Fabric environment"""
        
    except Exception as e:
        logger.error(f"Error generating Fabric code: {str(e)}")
        return f"Error generating Fabric code: {str(e)}"

@mcp.tool()
async def validate_fabric_code(
    code: str,
    ctx: Context = None,
) -> str:
    """Validate PySpark code for Microsoft Fabric compatibility and performance.

    Args:
        code: PySpark code to validate for Fabric compatibility
        ctx: Context object containing client information
    Returns:
        A string containing detailed validation results and Fabric-specific recommendations.
    """
    try:
        from helpers.pyspark_helpers import PySparkValidator
        
        validator = PySparkValidator()
        
        # Basic syntax validation
        validation_results = []
        try:
            compile(code, '<string>', 'exec')
            validation_results.append("✅ Syntax validation: PASSED")
        except SyntaxError as e:
            validation_results.append(f"❌ Syntax validation: FAILED - {e}")
            return "\n".join(validation_results)
        
        # Fabric compatibility checks
        fabric_results = validator.validate_fabric_compatibility(code)
        
        # Performance pattern checks
        performance_results = validator.check_performance_patterns(code)
        
        # Additional Fabric-specific checks
        fabric_warnings = []
        fabric_suggestions = []
        
        # Check for Fabric best practices
        if 'spark.table(' in code:
            validation_results.append("✅ Using Fabric managed tables")
        
        if 'notebookutils' in code:
            validation_results.append("✅ Using Fabric notebook utilities")
        
        if 'format("delta")' in code:
            validation_results.append("✅ Using Delta Lake format")
        
        # Check for potential issues
        if 'spark.sql("USE' in code:
            fabric_warnings.append("⚠️ Explicit USE statements may not be necessary in Fabric")
        
        if 'hdfs://' in code or 's3://' in code:
            fabric_warnings.append("⚠️ Direct file system paths detected - consider using Fabric's managed storage")
        
        # Compile comprehensive report
        result = "# Microsoft Fabric PySpark Code Validation Report\n\n"
        
        result += "## Basic Validation\n"
        result += "\n".join(validation_results) + "\n\n"
        
        if fabric_results["issues"]:
            result += "## Fabric Compatibility Issues\n"
            result += "\n".join(fabric_results["issues"]) + "\n\n"
        
        all_warnings = fabric_warnings + performance_results["warnings"]
        if all_warnings:
            result += "## Warnings\n"
            result += "\n".join(all_warnings) + "\n\n"
        
        all_suggestions = fabric_results["suggestions"] + fabric_suggestions + performance_results["optimizations"]
        if all_suggestions:
            result += "## Fabric Optimization Suggestions\n"
            result += "\n".join(all_suggestions) + "\n\n"
        
        # Summary
        total_issues = len(fabric_results["issues"])
        total_warnings = len(all_warnings)
        total_suggestions = len(all_suggestions)
        
        result += "## Summary\n"
        if total_issues == 0 and total_warnings == 0:
            result += "✅ Code is Fabric-ready! No critical issues detected.\n"
        else:
            result += f"📊 Found {total_issues} critical issues, {total_warnings} warnings, and {total_suggestions} optimization opportunities.\n"
        
        result += "\n### Fabric-Specific Recommendations:\n"
        result += "- Use `spark.table()` for managed tables in lakehouses\n"
        result += "- Leverage `notebookutils` for Fabric integration\n"
        result += "- Always use Delta Lake format for optimal performance\n"
        result += "- Consider partitioning strategies for large datasets\n"
        result += "- Use broadcast joins for dimension tables < 200MB\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating Fabric code: {str(e)}")
        return f"Error validating Fabric code: {str(e)}"

@mcp.tool()
async def analyze_notebook_performance(
    workspace: str,
    notebook_id: str,
    ctx: Context = None,
) -> str:
    """Analyze a notebook's code for performance optimization opportunities in Fabric.

    Args:
        workspace: Name or ID of the workspace
        notebook_id: ID or name of the notebook
        ctx: Context object containing client information
    Returns:
        A string containing performance analysis and optimization recommendations.
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        # Get notebook content
        notebook_content = await get_notebook_content(workspace, notebook_id, ctx)
        
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
            report += "✅ **Excellent** - Your notebook is well-optimized for Fabric!\n"
        elif score >= 60:
            report += "⚠️ **Good** - Some optimization opportunities exist.\n"
        elif score >= 40:
            report += "🔧 **Needs Improvement** - Several performance issues should be addressed.\n"
        else:
            report += "❌ **Poor** - Significant performance optimization required.\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Error analyzing notebook performance: {str(e)}")
        return f"Error analyzing notebook performance: {str(e)}"


@mcp.tool()
async def list_spark_jobs(
    workspace: Optional[str] = None,
    notebook: Optional[str] = None,
    state: Optional[str] = None,
    ctx: Context = None
) -> str:
    """List Spark job executions (Livy sessions).

    Lists Spark jobs running in a workspace or specific notebook. Jobs represent
    notebook executions and their current state.

    Args:
        workspace: Workspace name or ID (optional - uses context if not provided)
        notebook: Notebook name or ID (optional - if omitted, lists all workspace jobs)
        state: Filter by job state: NotStarted, InProgress, Cancelled, Failed, Succeeded (optional)
        ctx: Context object containing client information

    Returns:
        Formatted list of Spark jobs with status, runtime, and submitter information
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

        fabric_client = FabricApiClient(get_azure_credentials(ctx.client_id, __ctx_cache))
        spark_client = SparkClient(fabric_client)

        # Resolve workspace
        ws = workspace or __ctx_cache.get(f"{ctx.client_id}_workspace")
        if not ws:
            return "Workspace not set. Please set a workspace using 'set_workspace' command."

        workspace_id = await fabric_client.resolve_workspace(ws)

        # Build filters
        filters = {}
        if state:
            filters["state"] = state

        # Get sessions
        if notebook:
            # Resolve notebook ID if name provided
            if not _is_valid_uuid(notebook):
                notebook_id = await fabric_client.resolve_item_id(
                    item=notebook, type="Notebook", workspace=workspace_id
                )
            else:
                notebook_id = notebook

            sessions = await spark_client.list_notebook_sessions(workspace_id, notebook_id)

            # Apply client-side filtering for notebook sessions (API doesn't support it)
            if state and sessions:
                sessions = [s for s in sessions if s.get("state") == state]
        else:
            sessions = await spark_client.list_workspace_sessions(workspace_id, filters)

        if not sessions:
            filter_msg = f" with state '{state}'" if state else ""
            notebook_msg = f" for notebook '{notebook}'" if notebook else ""
            return f"No Spark jobs found{notebook_msg}{filter_msg} in workspace '{ws}'."

        # Format as markdown table
        markdown = f"# Spark Jobs in workspace '{ws}'\n\n"
        if notebook:
            markdown = f"# Spark Jobs for notebook '{notebook}' in workspace '{ws}'\n\n"
        if state:
            markdown += f"**Filtered by state:** {state}\n\n"

        markdown += "| Job ID | State | Submitted Time | Duration | Submitter |\n"
        markdown += "|--------|-------|----------------|----------|----------|\n"

        for session in sessions:
            job_id = session.get("livyId", "N/A")
            job_state = session.get("state", "Unknown")
            submitted_time = session.get("submittedDateTime", "N/A")
            submitter_id = session.get("submitter", {}).get("id", "N/A")

            # Calculate duration if available
            duration = "N/A"
            if submitted_time != "N/A":
                try:
                    submitted_dt = datetime.fromisoformat(submitted_time.replace("Z", "+00:00"))
                    end_time = session.get("endDateTime")
                    if end_time:
                        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                        duration_seconds = (end_dt - submitted_dt).total_seconds()
                        duration = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
                    elif job_state in ["InProgress", "NotStarted"]:
                        now_dt = datetime.now(submitted_dt.tzinfo)
                        duration_seconds = (now_dt - submitted_dt).total_seconds()
                        duration = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s (running)"
                except Exception:
                    duration = "N/A"

            # Add status emoji
            status_emoji = {
                "Succeeded": "✅",
                "Failed": "❌",
                "InProgress": "⏳",
                "NotStarted": "⏸️",
                "Cancelled": "🚫"
            }.get(job_state, "")

            markdown += f"| {job_id} | {status_emoji} {job_state} | {submitted_time} | {duration} | {submitter_id} |\n"

        markdown += f"\n**Total jobs:** {len(sessions)}\n"

        return markdown

    except Exception as e:
        logger.error(f"Error listing Spark jobs: {str(e)}")
        return f"Error listing Spark jobs: {str(e)}"


@mcp.tool()
async def get_job_details(
    job_id: str,
    workspace: Optional[str] = None,
    notebook: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get detailed information about a specific Spark job execution.

    Retrieves comprehensive details about a Spark job including execution status,
    timing information, error messages, input parameters, resource configuration,
    and links to monitoring tools.

    Args:
        job_id: Livy session ID (from list_spark_jobs)
        workspace: Workspace name or ID (optional - uses context if not provided)
        notebook: Notebook name or ID (required to get detailed logs)
        ctx: Context object containing client information

    Returns:
        Detailed job information including notebook name, error messages for failed jobs,
        input parameters, Spark application ID, and monitoring URLs
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")

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
            notebook_id = await fabric_client.resolve_item_id(
                item=notebook, type="Notebook", workspace=workspace_id
            )
        else:
            notebook_id = notebook

        # Get session details
        session = await spark_client.get_session_details(workspace_id, notebook_id, job_id)

        if not session:
            return f"No job found with ID '{job_id}' for notebook '{notebook}' in workspace '{ws}'."

        # Format detailed information
        markdown = f"# Spark Job Details\n\n"

        # Basic Information
        markdown += f"## Basic Information\n"
        markdown += f"**Job ID (Livy ID):** {session.get('livyId', 'N/A')}\n"
        markdown += f"**Workspace:** {ws}\n"

        # Notebook name from session
        notebook_name = session.get('itemName', notebook)
        markdown += f"**Notebook Name:** {notebook_name}\n"
        markdown += f"**Notebook ID:** {session.get('item', {}).get('id', notebook_id)}\n"

        # Job Instance ID for linking to DAG/Run
        job_instance_id = session.get('jobInstanceId')
        if job_instance_id:
            markdown += f"**Job Instance ID:** {job_instance_id}\n"

        markdown += f"**Job Type:** {session.get('jobType', 'N/A')}\n"
        markdown += f"**Operation:** {session.get('operationName', 'N/A')}\n\n"

        # Status information
        state = session.get('state', 'Unknown')
        status_emoji = {
            "Succeeded": "✅",
            "Failed": "❌",
            "InProgress": "⏳",
            "NotStarted": "⏸️",
            "Cancelled": "🚫"
        }.get(state, "")

        markdown += f"## Status\n"
        markdown += f"**State:** {status_emoji} {state}\n"

        # Cancellation reason if applicable
        if state == "Cancelled":
            cancellation_reason = session.get('cancellationReason')
            if cancellation_reason:
                markdown += f"**Cancellation Reason:** {cancellation_reason}\n"

        # Timing information with detailed durations
        submitted = session.get('submittedDateTime', 'N/A')
        started = session.get('startDateTime', 'N/A')
        ended = session.get('endDateTime', 'N/A')

        markdown += f"\n## Timing\n"
        markdown += f"**Submitted:** {submitted}\n"
        markdown += f"**Started:** {started}\n"
        markdown += f"**Ended:** {ended}\n"

        # Duration details from API
        queued_duration = session.get('queuedDuration')
        running_duration = session.get('runningDuration')
        total_duration = session.get('totalDuration')

        if queued_duration:
            markdown += f"**Queued Duration:** {queued_duration}\n"
        if running_duration:
            markdown += f"**Running Duration:** {running_duration}\n"
        if total_duration:
            markdown += f"**Total Duration:** {total_duration}\n"

        # Calculate duration if not provided by API
        if not total_duration and submitted != 'N/A' and ended != 'N/A':
            try:
                submitted_dt = datetime.fromisoformat(submitted.replace("Z", "+00:00"))
                ended_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                duration_seconds = (ended_dt - submitted_dt).total_seconds()
                markdown += f"**Calculated Duration:** {int(duration_seconds // 60)}m {int(duration_seconds % 60)}s\n"
            except Exception:
                pass

        # Submitter information
        submitter = session.get('submitter', {})
        markdown += f"\n## Submitter\n"
        markdown += f"**ID:** {submitter.get('id', 'N/A')}\n"
        markdown += f"**Type:** {submitter.get('type', 'N/A')}\n"

        # Resource Configuration
        markdown += f"\n## Resource Configuration\n"
        driver_memory = session.get('driverMemory')
        driver_cores = session.get('driverCores')
        executor_memory = session.get('executorMemory')
        executor_cores = session.get('executorCores')
        num_executors = session.get('numExecutors')

        if driver_memory or driver_cores:
            markdown += f"**Driver:** {driver_cores or 'N/A'} cores, {driver_memory or 'N/A'} GB memory\n"
        if executor_memory or executor_cores or num_executors:
            markdown += f"**Executors:** {num_executors or 'N/A'} executors × {executor_cores or 'N/A'} cores × {executor_memory or 'N/A'} GB memory\n"

        # Dynamic allocation
        is_dynamic = session.get('isDynamicAllocationEnabled')
        if is_dynamic:
            max_executors = session.get('dynamicAllocationMaxExecutors')
            markdown += f"**Dynamic Allocation:** Enabled (max {max_executors} executors)\n"

        # Runtime and environment
        runtime_version = session.get('runtimeVersion')
        if runtime_version:
            markdown += f"**Runtime Version:** {runtime_version}\n"

        # High concurrency mode
        is_high_concurrency = session.get('isHighConcurrency')
        if is_high_concurrency:
            markdown += f"**High Concurrency Mode:** Enabled\n"

        # Spark Application Information
        spark_app_id = session.get('sparkApplicationId')
        if spark_app_id:
            markdown += f"\n## Spark Application\n"
            markdown += f"**Spark Application ID:** {spark_app_id}\n"
            markdown += f"**Spark UI URL:** [View Spark UI](https://spark.fabric.microsoft.com/sparkui/{spark_app_id})\n"

        # Error information if failed - Enhanced with job instance details
        if state == "Failed":
            markdown += f"\n## Error Information\n"

            # Try to get detailed error from job instance API
            if job_instance_id:
                job_instance = await spark_client.get_job_instance(workspace_id, notebook_id, job_instance_id)
                if job_instance:
                    failure_reason = job_instance.get('failureReason')
                    if failure_reason:
                        markdown += f"**Failure Reason:** {failure_reason}\n\n"

                    # Include root activity ID for debugging
                    root_activity_id = job_instance.get('rootActivityId')
                    if root_activity_id:
                        markdown += f"**Root Activity ID:** {root_activity_id}\n"

            # Include any errors from the session response
            errors = session.get('errors', [])
            if errors:
                markdown += f"\n**Error Details:**\n"
                for error in errors:
                    error_code = error.get('errorCode', 'Unknown')
                    error_message = error.get('message', 'No message')
                    error_source = error.get('source', '')

                    markdown += f"- **{error_code}**"
                    if error_source:
                        markdown += f" (Source: {error_source})"
                    markdown += f"\n  {error_message}\n"

            # If no specific errors found, provide guidance
            if not errors and not (job_instance_id and job_instance and job_instance.get('failureReason')):
                markdown += "\n*No detailed error information available. Check the Spark logs for more details.*\n"

        # Additional metadata
        markdown += f"\n## Additional Information\n"
        markdown += f"**Capacity ID:** {session.get('capacityId', 'N/A')}\n"
        markdown += f"**Workspace ID:** {session.get('workspaceId', 'N/A')}\n"
        markdown += f"**Origin:** {session.get('origin', 'N/A')}\n"
        markdown += f"**Attempt Number:** {session.get('attemptNumber', 'N/A')} of {session.get('maxNumberOfAttempts', 'N/A')}\n"

        # Monitoring Links
        markdown += f"\n## Monitoring & Logs\n"
        markdown += f"- **Job Instance ID:** `{job_instance_id or 'N/A'}`\n"
        if spark_app_id:
            markdown += f"- **Spark Application UI:** [Open UI](https://spark.fabric.microsoft.com/sparkui/{spark_app_id})\n"
        markdown += f"- **Workspace:** [{ws}](https://app.fabric.microsoft.com/groups/{workspace_id})\n"
        markdown += f"- **Notebook:** [{notebook_name}](https://app.fabric.microsoft.com/groups/{workspace_id}/notebooks/{notebook_id})\n"

        return markdown

    except Exception as e:
        logger.error(f"Error getting job details: {str(e)}")
        return f"Error getting job details: {str(e)}"


# ===== AGENTIC / ORCHESTRATED TOOLS =====
# These tools use the orchestration layer for intelligent multi-step workflows

@mcp.tool()
async def create_notebook_validated(
    workspace: str,
    notebook_name: str,
    template_type: str = "basic",
    validate: bool = True,
    optimize: bool = False,
    ctx: Context = None
) -> str:
    """Create a PySpark notebook with automatic validation and optimization (AGENTIC).
    
    This is an intelligent orchestrated operation that:
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
        ctx: Context object
    
    Returns:
        JSON string with creation results, validation, and intelligent suggestions
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.notebook_orchestrator import notebook_orchestrator
        
        result = await notebook_orchestrator.create_validated_notebook(
            workspace=workspace,
            notebook_name=notebook_name,
            template_type=template_type,
            validate=validate,
            optimize=optimize,
            ctx=ctx
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in create_notebook_validated: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False
        }, indent=2)


@mcp.tool()
async def execute_notebook_intent(
    goal: str,
    workspace: Optional[str] = None,
    notebook_id: Optional[str] = None,
    execution_mode: str = "standard",
    ctx: Context = None
) -> str:
    """Execute notebook operations based on natural language intent (AGENTIC).
    
    Intelligent routing based on keywords in the goal:
    - Create/build → create_pyspark_notebook workflow
    - Validate/check → validate_pyspark_code workflow
    - Optimize/improve → analyze_notebook_performance + optimization workflow
    - Explore/analyze → comprehensive notebook analysis
    
    Execution modes:
    - fast: Quick preview, minimal validation (5x faster)
    - standard: Normal execution with validation (default)
    - analyze: Full analysis with performance metrics (detailed insights)
    - safe: Maximum validation with rollback capability (production-safe)
    
    Args:
        goal: Natural language description of what you want to achieve
        workspace: Fabric workspace name (optional, uses context if not provided)
        notebook_id: Notebook ID (for existing notebooks, optional)
        execution_mode: Execution mode (fast/standard/analyze/safe, default: standard)
        ctx: Context object
    
    Returns:
        JSON string with execution results and intelligent suggestions
    
    Examples:
        - "Create a new ETL notebook and validate it"
        - "Optimize my slow notebook"
        - "Analyze the performance of my notebook"
        - "Validate the notebook code for errors"
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.agent_policy import agent_policy
        from helpers.utils.context import get_context
        
        # Get context
        ctx_obj = get_context()
        context = {
            'workspace': workspace or ctx_obj.workspace,
            'notebook_id': notebook_id,
        }
        
        # Execute intent
        result = await agent_policy.execute_intent(
            intent=goal,
            domain='notebook',
            context=context,
            execution_mode=execution_mode,
            ctx=ctx
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in execute_notebook_intent: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False,
            "suggestion": "Try rephrasing your goal or provide more specific details"
        }, indent=2)


@mcp.tool()
async def get_notebook_suggestions(
    notebook_id: str,
    workspace: str,
    ctx: Context = None
) -> str:
    """Get intelligent next-step suggestions for a notebook (AGENTIC).
    
    Analyzes the notebook and provides context-aware suggestions for:
    - Code improvements
    - Performance optimizations
    - Best practices
    - Next logical steps
    
    Args:
        notebook_id: Notebook ID
        workspace: Workspace name
        ctx: Context object
    
    Returns:
        JSON string with intelligent suggestions
    """
    try:
        if ctx is None:
            raise ValueError("Context (ctx) must be provided.")
        
        from helpers.orchestration.notebook_orchestrator import notebook_orchestrator
        
        result = await notebook_orchestrator.analyze_notebook_comprehensive(
            notebook_id=notebook_id,
            workspace=workspace,
            ctx=ctx
        )
        
        # Extract just the suggestions
        return json.dumps({
            "notebook_id": notebook_id,
            "suggestions": result.get('suggested_next_actions', []),
            "summary": result.get('summary', {}),
            "workflow": "get_notebook_suggestions"
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error in get_notebook_suggestions: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False
        }, indent=2)


@mcp.tool()
async def list_available_workflows(ctx: Context = None) -> str:
    """List all available pre-defined workflows (AGENTIC).
    
    Returns information about intelligent workflow chains that can automate
    multi-step operations like:
    - Complete notebook development
    - Lakehouse data exploration
    - ETL pipeline setup
    - ML notebook setup
    - Performance optimization pipelines
    
    Returns:
        JSON string with available workflows and their descriptions
    """
    try:
        from helpers.orchestration.agent_policy import agent_policy
        
        workflows = agent_policy.get_available_workflows()
        
        return json.dumps(workflows, indent=2)
    
    except Exception as e:
        logger.error(f"Error in list_available_workflows: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False
        }, indent=2)


