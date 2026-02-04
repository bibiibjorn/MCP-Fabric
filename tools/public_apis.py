"""
Public APIs tools for Microsoft Fabric MCP Server.

Provides access to OpenAPI specifications, best practices documentation,
item definitions, and API examples for Microsoft Fabric workloads.
"""

import json
import os
from pathlib import Path
from typing import Optional

from helpers.logging_config import get_logger
from helpers.utils.context import mcp

logger = get_logger(__name__)

# Get the resources directory path
RESOURCES_DIR = Path(__file__).parent.parent / "resources"
BEST_PRACTICES_DIR = RESOURCES_DIR / "best-practices"
ITEM_DEFINITIONS_DIR = RESOURCES_DIR / "item-definitions"
OPENAPI_SPECS_DIR = RESOURCES_DIR / "openapi-specs"


def _get_available_workloads() -> list[str]:
    """Get list of available workload types from OpenAPI specs directory."""
    if not OPENAPI_SPECS_DIR.exists():
        return []
    return sorted([
        d.name for d in OPENAPI_SPECS_DIR.iterdir()
        if d.is_dir() and d.name != "common"
    ])


def _get_available_topics() -> list[str]:
    """Get list of available best practices topics."""
    if not BEST_PRACTICES_DIR.exists():
        return []
    return sorted([
        f.stem for f in BEST_PRACTICES_DIR.glob("*.md")
    ])


def _normalize_workload_type(workload_type: str) -> str:
    """Normalize workload type to match folder naming conventions."""
    # Map common aliases
    aliases = {
        "common": "platform",
        "semantic-model": "semanticModel",
        "semantic_model": "semanticModel",
        "kql-database": "kqlDatabase",
        "kql_database": "kqlDatabase",
        "data-pipeline": "dataPipeline",
        "data_pipeline": "dataPipeline",
    }
    return aliases.get(workload_type.lower(), workload_type)


@mcp.tool()
async def list_fabric_workloads() -> str:
    """List all Microsoft Fabric workload types that have public API specifications available.

    Returns a list of workload types (e.g., lakehouse, notebook, semanticModel) that
    have OpenAPI specifications bundled with this server.

    Returns:
        A formatted list of available workload types.
    """
    workloads = _get_available_workloads()

    if not workloads:
        return "No workload specifications found. Please ensure resources are properly installed."

    result = "# Available Microsoft Fabric Workload Types\n\n"
    result += f"Found {len(workloads)} workload types with API specifications:\n\n"

    for workload in workloads:
        result += f"- `{workload}`\n"

    result += "\n## Usage\n\n"
    result += "Use `get_fabric_openapi_spec(workload_type='<workload>')` to retrieve the OpenAPI specification for a specific workload.\n"
    result += "Use `get_fabric_item_definition(workload_type='<workload>')` to get the item definition schema.\n"
    result += "Use `get_fabric_api_examples(workload_type='<workload>')` to get example API requests/responses.\n"

    return result


@mcp.tool()
async def get_fabric_openapi_spec(workload_type: str) -> str:
    """Retrieve the complete OpenAPI/Swagger specification for a specific Microsoft Fabric workload.

    Args:
        workload_type: The type of Fabric workload (e.g., 'lakehouse', 'notebook', 'semanticModel').
                       Use list_fabric_workloads() to see available types.

    Returns:
        The OpenAPI specification as a JSON string, or an error message if not found.
    """
    workload = _normalize_workload_type(workload_type)
    spec_dir = OPENAPI_SPECS_DIR / workload
    swagger_path = spec_dir / "swagger.json"

    if not swagger_path.exists():
        available = _get_available_workloads()
        return (
            f"Error: OpenAPI specification not found for workload '{workload_type}'.\n\n"
            f"Available workloads: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n\n"
            "Tip: Use list_fabric_workloads() to see all available workload types."
        )

    try:
        with open(swagger_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        # Also load definitions if they exist
        definitions_path = spec_dir / "definitions.json"
        if definitions_path.exists():
            with open(definitions_path, "r", encoding="utf-8") as f:
                definitions = json.load(f)
            spec["_bundled_definitions"] = definitions

        result = f"# OpenAPI Specification for {workload_type}\n\n"
        result += f"**Title**: {spec.get('info', {}).get('title', 'N/A')}\n"
        result += f"**Version**: {spec.get('info', {}).get('version', 'N/A')}\n"
        result += f"**Host**: {spec.get('host', 'N/A')}\n"
        result += f"**Base Path**: {spec.get('basePath', 'N/A')}\n\n"

        # List available paths/operations
        paths = spec.get("paths", {})
        result += f"## Available Endpoints ({len(paths)} paths)\n\n"
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    summary = details.get("summary", "No summary")
                    result += f"- **{method.upper()}** `{path}`\n  {summary}\n"

        result += "\n## Full Specification (JSON)\n\n```json\n"
        result += json.dumps(spec, indent=2)
        result += "\n```"

        return result

    except json.JSONDecodeError as e:
        return f"Error: Failed to parse OpenAPI specification: {e}"
    except Exception as e:
        logger.error(f"Error reading OpenAPI spec: {e}")
        return f"Error: Failed to read OpenAPI specification: {e}"


@mcp.tool()
async def get_fabric_platform_api() -> str:
    """Retrieve the OpenAPI/Swagger specification for Microsoft Fabric platform-level APIs.

    Platform APIs include core operations like workspaces, items, long-running operations,
    and other cross-workload functionality.

    Returns:
        The platform API OpenAPI specification.
    """
    return await get_fabric_openapi_spec("platform")


@mcp.tool()
async def get_fabric_best_practices(topic: str) -> str:
    """Retrieve best practice documentation and guidance for a specific Microsoft Fabric topic.

    Available topics:
    - throttling: API throttling and rate limiting best practices
    - admin-apis: Guidelines for using Admin APIs vs Standard APIs
    - pagination: Pagination patterns for Fabric REST APIs
    - long-running-operation: Handling async/LRO patterns

    Args:
        topic: The best practices topic to retrieve.

    Returns:
        The best practices documentation content.
    """
    available_topics = _get_available_topics()

    # Try exact match first
    topic_path = BEST_PRACTICES_DIR / f"{topic}.md"

    # Try with different separators
    if not topic_path.exists():
        normalized = topic.lower().replace("_", "-").replace(" ", "-")
        topic_path = BEST_PRACTICES_DIR / f"{normalized}.md"

    if not topic_path.exists():
        return (
            f"Error: Best practices topic '{topic}' not found.\n\n"
            f"Available topics:\n"
            + "\n".join(f"- `{t}`" for t in available_topics)
            + "\n\nTip: Use exact topic names as listed above."
        )

    try:
        with open(topic_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading best practices: {e}")
        return f"Error: Failed to read best practices documentation: {e}"


@mcp.tool()
async def list_fabric_best_practices() -> str:
    """List all available best practices topics for Microsoft Fabric development.

    Returns:
        A list of available best practices topics with descriptions.
    """
    topics = _get_available_topics()

    if not topics:
        return "No best practices documentation found."

    result = "# Microsoft Fabric Best Practices\n\n"
    result += "The following best practices documentation is available:\n\n"

    topic_descriptions = {
        "throttling": "API throttling and rate limiting - how to handle 429 responses with retry logic",
        "admin-apis": "Guidelines for when to use Admin APIs vs Standard APIs, permission requirements",
        "pagination": "How to handle paginated responses using continuationToken",
        "long-running-operation": "Handling asynchronous long-running operations (LRO) with polling",
    }

    for topic in topics:
        desc = topic_descriptions.get(topic, "Documentation available")
        result += f"- **{topic}**: {desc}\n"

    result += "\n## Usage\n\n"
    result += "Use `get_fabric_best_practices(topic='<topic>')` to retrieve the full documentation.\n"

    return result


@mcp.tool()
async def get_fabric_item_definition(workload_type: str) -> str:
    """Retrieve the JSON schema definition for a specific Microsoft Fabric item type.

    Item definitions describe the structure and properties of Fabric items like
    Lakehouses, Notebooks, Pipelines, etc.

    Args:
        workload_type: The type of Fabric item (e.g., 'lakehouse', 'notebook', 'semantic-model').

    Returns:
        The item definition documentation including schema, examples, and usage.
    """
    # Map workload types to definition file names
    file_mappings = {
        "lakehouse": "lakehouse-definition.md",
        "notebook": "notebook-definition.md",
        "semanticmodel": "semantic-model-definition.md",
        "semantic-model": "semantic-model-definition.md",
        "semantic_model": "semantic-model-definition.md",
        "datapipeline": "datapipeline-definition.md",
        "data-pipeline": "datapipeline-definition.md",
        "data_pipeline": "datapipeline-definition.md",
        "dataflow": "dataflow-definition.md",
        "report": "report-definition.md",
        "environment": "environment-definition.md",
        "kqldatabase": "kql-database-definition.md",
        "kql-database": "kql-database-definition.md",
        "kql_database": "kql-database-definition.md",
        "eventhouse": "eventhouse-definition.md",
        "eventstream": "eventstream-definition.md",
        "reflex": "reflex-definition.md",
        "ontology": "ontology-definition.md",
        "sparkjobdefinition": "spark-job-definition.md",
        "spark-job-definition": "spark-job-definition.md",
        "graphqlapi": "graphql-api-definition.md",
        "graphql-api": "graphql-api-definition.md",
        "copyjob": "copyjob-definition.md",
        "copy-job": "copyjob-definition.md",
        "mirroreddatabase": "mirrored-database-definition.md",
        "mirrored-database": "mirrored-database-definition.md",
        "warehouse": "warehouse-definition.md",  # Note: may not exist
    }

    normalized = workload_type.lower().replace(" ", "")
    filename = file_mappings.get(normalized)

    # If not in mappings, try to find by pattern
    if not filename:
        # Try direct match with -definition.md suffix
        possible_names = [
            f"{workload_type.lower()}-definition.md",
            f"{workload_type.lower().replace('_', '-')}-definition.md",
        ]
        for name in possible_names:
            if (ITEM_DEFINITIONS_DIR / name).exists():
                filename = name
                break

    if not filename:
        # List available definitions
        available = sorted([
            f.stem.replace("-definition", "")
            for f in ITEM_DEFINITIONS_DIR.glob("*-definition.md")
        ])
        return (
            f"Error: Item definition not found for '{workload_type}'.\n\n"
            f"Available item definitions:\n"
            + "\n".join(f"- `{d}`" for d in available[:15])
            + (f"\n... and {len(available) - 15} more" if len(available) > 15 else "")
        )

    definition_path = ITEM_DEFINITIONS_DIR / filename

    if not definition_path.exists():
        return f"Error: Item definition file not found: {filename}"

    try:
        with open(definition_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading item definition: {e}")
        return f"Error: Failed to read item definition: {e}"


@mcp.tool()
async def list_fabric_item_definitions() -> str:
    """List all available item definition schemas for Microsoft Fabric.

    Returns:
        A list of available item types with definition schemas.
    """
    if not ITEM_DEFINITIONS_DIR.exists():
        return "No item definitions found."

    definitions = sorted([
        f.stem.replace("-definition", "")
        for f in ITEM_DEFINITIONS_DIR.glob("*-definition.md")
    ])

    if not definitions:
        return "No item definitions found."

    result = "# Microsoft Fabric Item Definitions\n\n"
    result += f"Found {len(definitions)} item definition schemas:\n\n"

    for defn in definitions:
        result += f"- `{defn}`\n"

    result += "\n## Usage\n\n"
    result += "Use `get_fabric_item_definition(workload_type='<type>')` to retrieve the full schema documentation.\n"
    result += "\nItem definitions describe:\n"
    result += "- Supported formats for the item type\n"
    result += "- Definition parts and their structure\n"
    result += "- JSON schema for payloads\n"
    result += "- Example configurations\n"

    return result


@mcp.tool()
async def get_fabric_api_examples(workload_type: str) -> str:
    """Retrieve example API request/response files for a specific Microsoft Fabric workload.

    Examples show real request bodies and expected responses for API operations.

    Args:
        workload_type: The type of Fabric workload (e.g., 'lakehouse', 'notebook').

    Returns:
        A collection of example API calls with request/response payloads.
    """
    workload = _normalize_workload_type(workload_type)
    examples_dir = OPENAPI_SPECS_DIR / workload / "examples"

    if not examples_dir.exists():
        available = [
            d.name for d in OPENAPI_SPECS_DIR.iterdir()
            if d.is_dir() and (d / "examples").exists()
        ]
        return (
            f"Error: No examples found for workload '{workload_type}'.\n\n"
            f"Workloads with examples: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}"
        )

    try:
        examples = {}

        def collect_examples(directory: Path, prefix: str = ""):
            """Recursively collect example files."""
            for item in directory.iterdir():
                if item.is_file() and item.suffix == ".json":
                    key = f"{prefix}{item.stem}" if prefix else item.stem
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            examples[key] = json.load(f)
                    except json.JSONDecodeError:
                        examples[key] = {"error": "Failed to parse JSON"}
                elif item.is_dir():
                    collect_examples(item, f"{prefix}{item.name}/")

        collect_examples(examples_dir)

        if not examples:
            return f"No example files found for workload '{workload_type}'."

        result = f"# API Examples for {workload_type}\n\n"
        result += f"Found {len(examples)} example files:\n\n"

        for name, content in sorted(examples.items()):
            result += f"## {name}\n\n```json\n"
            result += json.dumps(content, indent=2)
            result += "\n```\n\n"

        return result

    except Exception as e:
        logger.error(f"Error reading API examples: {e}")
        return f"Error: Failed to read API examples: {e}"
