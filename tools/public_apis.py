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


# File mappings for item definitions
_ITEM_DEFINITION_MAPPINGS = {
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
    "warehouse": "warehouse-definition.md",
}

# Topic descriptions for best practices listing
_TOPIC_DESCRIPTIONS = {
    "throttling": "API throttling and rate limiting - how to handle 429 responses with retry logic",
    "admin-apis": "Guidelines for when to use Admin APIs vs Standard APIs, permission requirements",
    "pagination": "How to handle paginated responses using continuationToken",
    "long-running-operation": "Handling asynchronous long-running operations (LRO) with polling",
}


@mcp.tool()
async def fabric_reference(
    action: str,
    workload_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> str:
    """Access Microsoft Fabric API specifications, best practices, item definitions, and examples.

    Provides reference documentation bundled with this server for Microsoft Fabric
    workloads, including OpenAPI specs, best practices guides, item definition schemas,
    and example API requests/responses.

    Args:
        action: Operation to perform:
            'list_workloads' - List all available Fabric workload types with API specs
            'get_openapi_spec' - Get the OpenAPI specification for a workload
            'get_platform_api' - Get the platform-level API specification
            'get_best_practices' - Get best practices documentation for a topic
            'list_best_practices' - List all available best practices topics
            'get_item_definition' - Get the item definition schema for a workload type
            'list_item_definitions' - List all available item definition schemas
            'get_api_examples' - Get example API requests/responses for a workload
        workload_type: Fabric workload type (required for get_openapi_spec, get_item_definition, get_api_examples).
            Examples: 'lakehouse', 'notebook', 'semanticModel', 'dataPipeline'
        topic: Best practices topic (required for get_best_practices).
            Examples: 'throttling', 'admin-apis', 'pagination', 'long-running-operation'

    Returns:
        Reference documentation, specifications, or listing of available resources
    """
    try:
        # --- list_workloads ---
        if action == "list_workloads":
            workloads = _get_available_workloads()

            if not workloads:
                return "No workload specifications found. Please ensure resources are properly installed."

            result = "# Available Microsoft Fabric Workload Types\n\n"
            result += f"Found {len(workloads)} workload types with API specifications:\n\n"

            for workload in workloads:
                result += f"- `{workload}`\n"

            result += "\n## Usage\n\n"
            result += "Use `fabric_reference(action='get_openapi_spec', workload_type='<workload>')` to retrieve the OpenAPI specification for a specific workload.\n"
            result += "Use `fabric_reference(action='get_item_definition', workload_type='<workload>')` to get the item definition schema.\n"
            result += "Use `fabric_reference(action='get_api_examples', workload_type='<workload>')` to get example API requests/responses.\n"

            return result

        # --- get_openapi_spec ---
        elif action == "get_openapi_spec":
            if not workload_type:
                return "Error: workload_type is required for 'get_openapi_spec' action."

            workload = _normalize_workload_type(workload_type)
            spec_dir = OPENAPI_SPECS_DIR / workload
            swagger_path = spec_dir / "swagger.json"

            if not swagger_path.exists():
                available = _get_available_workloads()
                return (
                    f"Error: OpenAPI specification not found for workload '{workload_type}'.\n\n"
                    f"Available workloads: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n\n"
                    "Tip: Use fabric_reference(action='list_workloads') to see all available workload types."
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

        # --- get_platform_api ---
        elif action == "get_platform_api":
            # Delegate to get_openapi_spec with platform workload type
            workload = "platform"
            spec_dir = OPENAPI_SPECS_DIR / workload
            swagger_path = spec_dir / "swagger.json"

            if not swagger_path.exists():
                available = _get_available_workloads()
                return (
                    f"Error: OpenAPI specification not found for platform APIs.\n\n"
                    f"Available workloads: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n\n"
                    "Tip: Use fabric_reference(action='list_workloads') to see all available workload types."
                )

            try:
                with open(swagger_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)

                definitions_path = spec_dir / "definitions.json"
                if definitions_path.exists():
                    with open(definitions_path, "r", encoding="utf-8") as f:
                        definitions = json.load(f)
                    spec["_bundled_definitions"] = definitions

                result = f"# OpenAPI Specification for platform\n\n"
                result += f"**Title**: {spec.get('info', {}).get('title', 'N/A')}\n"
                result += f"**Version**: {spec.get('info', {}).get('version', 'N/A')}\n"
                result += f"**Host**: {spec.get('host', 'N/A')}\n"
                result += f"**Base Path**: {spec.get('basePath', 'N/A')}\n\n"

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

        # --- get_best_practices ---
        elif action == "get_best_practices":
            if not topic:
                return "Error: topic is required for 'get_best_practices' action."

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

        # --- list_best_practices ---
        elif action == "list_best_practices":
            topics = _get_available_topics()

            if not topics:
                return "No best practices documentation found."

            result = "# Microsoft Fabric Best Practices\n\n"
            result += "The following best practices documentation is available:\n\n"

            for t in topics:
                desc = _TOPIC_DESCRIPTIONS.get(t, "Documentation available")
                result += f"- **{t}**: {desc}\n"

            result += "\n## Usage\n\n"
            result += "Use `fabric_reference(action='get_best_practices', topic='<topic>')` to retrieve the full documentation.\n"

            return result

        # --- get_item_definition ---
        elif action == "get_item_definition":
            if not workload_type:
                return "Error: workload_type is required for 'get_item_definition' action."

            normalized = workload_type.lower().replace(" ", "")
            filename = _ITEM_DEFINITION_MAPPINGS.get(normalized)

            # If not in mappings, try to find by pattern
            if not filename:
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

        # --- list_item_definitions ---
        elif action == "list_item_definitions":
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
            result += "Use `fabric_reference(action='get_item_definition', workload_type='<type>')` to retrieve the full schema documentation.\n"
            result += "\nItem definitions describe:\n"
            result += "- Supported formats for the item type\n"
            result += "- Definition parts and their structure\n"
            result += "- JSON schema for payloads\n"
            result += "- Example configurations\n"

            return result

        # --- get_api_examples ---
        elif action == "get_api_examples":
            if not workload_type:
                return "Error: workload_type is required for 'get_api_examples' action."

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

        else:
            return f"Error: Unknown action '{action}'. Use 'list_workloads', 'get_openapi_spec', 'get_platform_api', 'get_best_practices', 'list_best_practices', 'get_item_definition', 'list_item_definitions', or 'get_api_examples'."

    except Exception as e:
        return f"Error in fabric reference: {str(e)}"
