from mcp.server.fastmcp import FastMCP
from cachetools import TTLCache


# Create MCP instance with context manager
mcp = FastMCP(
    "Fabric MCP Server",
    instructions="""You are a Microsoft Fabric assistant. Always use the provided tools to answer questions about Fabric resources.

Key behaviors:
- When asked to show, display, or read notebook code/content, call manage_notebook with action="get_content" and return the result directly. Do NOT use artifacts, skills, or any other mechanism — just display the tool output as-is.
- When asked about data, tables, or queries, use the appropriate tool (run_query, query_semantic_model, manage_tables, etc.) and return the results directly.
- Never read external skill files or documentation to answer questions that can be answered by calling a tool.
- Always prefer calling a tool over generating code or explanations from your own knowledge.
- Return tool results as-is in markdown format. Do not wrap them in code artifacts unless the user explicitly asks for an artifact.""",
    json_response=True,
    stateless_http=True,
)
mcp.settings.log_level = "debug"

# Shared cache and context
__ctx_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour


class ContextWrapper:
    """Wrapper for context that provides attribute access to cached values."""

    def __init__(self, cache: TTLCache, client_id: str = "default"):
        self._cache = cache
        self._client_id = client_id

    @property
    def workspace(self):
        return self._cache.get(f"{self._client_id}_workspace")

    @property
    def lakehouse(self):
        return self._cache.get(f"{self._client_id}_lakehouse")

    @property
    def semantic_model(self):
        return self._cache.get(f"{self._client_id}_semantic_model")

    @property
    def semantic_model_name(self):
        return self._cache.get(f"{self._client_id}_semantic_model_name")


def get_context(client_id: str = "default") -> ContextWrapper:
    """Get a context wrapper for accessing cached values.

    Args:
        client_id: The client ID to use for cache keys (default: "default")

    Returns:
        ContextWrapper with properties for workspace, lakehouse, semantic_model
    """
    return ContextWrapper(__ctx_cache, client_id)
