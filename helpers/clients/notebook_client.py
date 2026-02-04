from helpers.utils import _is_valid_uuid
from helpers.logging_config import get_logger
from helpers.clients.fabric_client import FabricApiClient
from typing import Dict, Any

logger = get_logger(__name__)


class NotebookClient:
    def __init__(self, client: FabricApiClient):
        self.client = client

    async def list_notebooks(self, workspace: str):
        """List all notebooks in a workspace."""
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")
        notebooks = await self.client.get_notebooks(workspace)

        if not notebooks:
            return f"No notebooks found in workspace '{workspace}'."

        markdown = f"# Notebooks in workspace '{workspace}'\n\n"
        markdown += "| ID | Name |\n"
        markdown += "|-----|------|\n"

        for nb in notebooks:
            markdown += f"| {nb['id']} | {nb['displayName']} |\n"

        return markdown

    async def get_notebook(self, workspace: str, notebook_id: str) -> Dict[str, Any]:
        """Get a specific notebook by ID."""
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")
        if not _is_valid_uuid(notebook_id):
            raise ValueError("Invalid notebook ID.")

        notebook = await self.client.get_notebook(workspace, notebook_id)

        if not notebook:
            return (
                f"No notebook found with ID '{notebook_id}' in workspace '{workspace}'."
            )

        return notebook

    async def get_notebook_definition(self, workspace: str, notebook_id: str) -> Dict[str, Any]:
        """Get the definition (content) of a specific notebook.

        This method calls the getDefinition endpoint which returns the actual
        notebook content including the .ipynb payload with all cells and code.

        Args:
            workspace: ID of the workspace (must be a valid UUID)
            notebook_id: ID of the notebook (must be a valid UUID)

        Returns:
            Dictionary containing the notebook definition with parts including the ipynb content,
            or an error message string if the notebook is not found.
        """
        if not _is_valid_uuid(workspace):
            raise ValueError("Invalid workspace ID.")
        if not _is_valid_uuid(notebook_id):
            raise ValueError("Invalid notebook ID.")

        definition = await self.client.get_notebook_definition(workspace, notebook_id)

        if not definition:
            return (
                f"No notebook definition found for ID '{notebook_id}' in workspace '{workspace}'."
            )

        return definition

    async def create_notebook(
        self, workspace: str, notebook_name: str, content: str
    ) -> Dict[str, Any]:
        """Create a new notebook."""
        try:
            workspace, workspace_id = await self.client.resolve_workspace_name_and_id(
                workspace
            )
            if not workspace_id:
                raise ValueError("Invalid workspace ID.")
            
            logger.info(f"Creating notebook '{notebook_name}' in workspace '{workspace}' (ID: {workspace_id}).")
            
            try:
                response = await self.client.create_notebook(
                    workspace_id=workspace_id,
                    notebook_name=notebook_name,
                    ipynb_name=notebook_name,
                    content=content,
                )
            except Exception as e:
                error_msg = f"Failed to create notebook '{notebook_name}' in workspace '{workspace}': {str(e)}"
                logger.error(error_msg)
                return error_msg

            
            logger.info(f"Successfully created notebook '{notebook_name}' with ID: {response['id']}")
            return response
            
        except Exception as e:
            error_msg = f"Error creating notebook '{notebook_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg
