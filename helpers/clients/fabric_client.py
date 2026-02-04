from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple, Union
import base64
from urllib.parse import quote
from functools import lru_cache
import requests
from azure.identity import DefaultAzureCredential
from helpers.logging_config import get_logger
from helpers.utils import _is_valid_uuid
import json
from uuid import UUID

logger = get_logger(__name__)
# from  sempy_labs._helper_functions import create_item



class FabricApiConfig(BaseModel):
    """Configuration for Fabric API"""

    base_url: str = "https://api.fabric.microsoft.com/v1"
    max_results: int = 100
    # Throttling/retry configuration (following Microsoft best practices)
    max_retries: int = 3
    default_retry_after: int = 60  # Default wait time if no Retry-After header
    enable_exponential_backoff: bool = True


class FabricApiClient:
    """Client for communicating with the Fabric API"""

    def __init__(self, credential=None, config=None):
        self.credential = credential or DefaultAzureCredential()
        self.config = config or FabricApiConfig()
        # Initialize cached methods
        self._cached_resolve_workspace = lru_cache(maxsize=128)(self._resolve_workspace)
        self._cached_resolve_lakehouse = lru_cache(maxsize=128)(self._resolve_lakehouse)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Fabric API calls"""
        return {
            "Authorization": f"Bearer {self.credential.get_token('https://api.fabric.microsoft.com/.default').token}"
        }

    def _build_url(
        self, endpoint: str, continuation_token: Optional[str] = None
    ) -> str:
        # If the endpoint starts with http, use it as-is.
        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{self.config.base_url}/{endpoint.lstrip('/')}"
        )
        if continuation_token:
            separator = "&" if "?" in url else "?"
            encoded_token = quote(continuation_token)
            url += f"{separator}continuationToken={encoded_token}"
        return url

    def _execute_with_retry(
        self,
        request_func,
        max_retries: Optional[int] = None,
    ) -> requests.Response:
        """
        Execute a request function with automatic retry on throttling (429 responses).

        Implements Microsoft Fabric API best practices:
        - Respects Retry-After header
        - Uses exponential backoff for subsequent retries
        - Logs throttling events for monitoring

        Args:
            request_func: A callable that returns a requests.Response
            max_retries: Maximum number of retries (defaults to config value)

        Returns:
            The successful response

        Raises:
            requests.RequestException: If all retries are exhausted
        """
        import time

        max_retries = max_retries or self.config.max_retries
        retry_count = 0

        while retry_count <= max_retries:
            response = request_func()

            if response.status_code == 429:  # Too Many Requests (Throttled)
                if retry_count >= max_retries:
                    logger.error(
                        f"Request failed after {max_retries} retries due to throttling"
                    )
                    response.raise_for_status()

                # Get Retry-After header (defaults to configured value)
                retry_after = int(
                    response.headers.get("Retry-After", self.config.default_retry_after)
                )

                # Apply exponential backoff if enabled
                if self.config.enable_exponential_backoff and retry_count > 0:
                    # Exponential: 2^retry_count * base_delay, capped at retry_after * 2
                    backoff_multiplier = min(2 ** retry_count, 4)
                    retry_after = min(retry_after * backoff_multiplier, retry_after * 2)

                logger.warning(
                    f"API throttled (429). Waiting {retry_after}s before retry "
                    f"{retry_count + 1}/{max_retries}. "
                    f"Consider reducing request frequency."
                )

                time.sleep(retry_after)
                retry_count += 1
                continue

            # For non-throttling responses, return immediately
            return response

        # Should not reach here, but just in case
        return response

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        use_pagination: bool = False,
        data_key: str = "value",
        lro: bool = False,
        lro_poll_interval: int = 2,  # seconds between polls
        lro_timeout: int = 300,  # max seconds to wait
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make an asynchronous call to the Fabric API.

        If use_pagination is True, it will automatically handle paginated responses.

        If lro is True, will poll for long-running operation completion.
        """
        import time

        params = params or {}

        if not use_pagination:
            url = self._build_url(endpoint=endpoint)
            try:
                if method.upper() == "POST":
                    # Use retry wrapper for throttling protection
                    response = self._execute_with_retry(
                        lambda: requests.post(
                            url,
                            headers=self._get_headers(),
                            json=params,
                            timeout=120,
                        )
                    )
                else:
                    if "maxResults" not in params:
                        params["maxResults"] = self.config.max_results
                    # Use retry wrapper for throttling protection
                    response = self._execute_with_retry(
                        lambda url=url, params=params: requests.request(
                            method=method.upper(),
                            url=url,
                            headers=self._get_headers(),
                            params=params,
                            timeout=120,
                        )
                    )
    
                # LRO support: check for 202 and Operation-Location or Location header
                if lro and response.status_code == 202:
                    # Fabric API uses different headers for LRO
                    op_url = (
                        response.headers.get("Operation-Location")
                        or response.headers.get("operation-location")
                        or response.headers.get("Location")
                        or response.headers.get("location")
                    )
                    if not op_url:
                        # Some endpoints return 202 with Retry-After but result in body
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            logger.info(f"LRO: Retry-After header found, waiting {retry_after}s...")
                            time.sleep(int(retry_after))
                            # Retry the same request
                            response = requests.post(
                                url,
                                headers=self._get_headers(),
                                json=params,
                                timeout=120,
                            )
                            if response.status_code == 200:
                                return response.json()
                        logger.error("LRO: No Operation-Location or Location header found.")
                        logger.debug(f"LRO: Response headers: {dict(response.headers)}")
                        return None
                    logger.info(f"LRO: Polling {op_url} for operation status...")
                    start_time = time.time()
                    while True:
                        # Use retry wrapper for LRO polling (throttling protection)
                        poll_resp = self._execute_with_retry(
                            lambda: requests.get(
                                op_url, headers=self._get_headers(), timeout=60
                            )
                        )
                        if poll_resp.status_code not in (200, 201, 202):
                            logger.error(
                                f"LRO: Poll failed with status {poll_resp.status_code}"
                            )
                            return None
                        poll_data = poll_resp.json()
                        status = poll_data.get("status") or poll_data.get(
                            "operationStatus"
                        )
                        if status in (
                            "Succeeded",
                            "succeeded",
                            "Completed",
                            "completed",
                        ):
                            logger.info("LRO: Operation succeeded.")
                            # Check if there's a result URL to fetch the actual data
                            # The result is at /operations/{operationId}/result
                            if "/operations/" in op_url:
                                result_url = op_url.rstrip("/") + "/result"
                                logger.info(f"LRO: Fetching result from {result_url}")
                                # Use retry wrapper for result fetch (throttling protection)
                                result_resp = self._execute_with_retry(
                                    lambda: requests.get(
                                        result_url, headers=self._get_headers(), timeout=60
                                    )
                                )
                                if result_resp.status_code == 200:
                                    return result_resp.json()
                                else:
                                    logger.warning(
                                        f"LRO: Result fetch returned {result_resp.status_code}, returning poll data"
                                    )
                            return poll_data
                        if status in ("Failed", "failed", "Canceled", "canceled"):
                            logger.error(
                                f"LRO: Operation failed or canceled. Status: {status}"
                            )
                            return poll_data
                        if time.time() - start_time > lro_timeout:
                            logger.error("LRO: Polling timed out.")
                            return None
                        logger.debug(
                            f"LRO: Status {status}, waiting {lro_poll_interval}s..."
                        )
                        time.sleep(lro_poll_interval)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.error(f"API call failed: {str(e)}")
                if e.response is not None:
                    logger.error(f"Response content: {e.response.text}")
                return None
        else:
            # Paginated request - implements Microsoft Fabric pagination best practices
            # using continuationToken pattern
            results = []
            continuation_token = None
            while True:
                url = self._build_url(
                    endpoint=endpoint, continuation_token=continuation_token
                )
                request_params = params.copy()
                # Remove any existing continuationToken in parameters to avoid conflict.
                request_params.pop("continuationToken", None)
                try:
                    if method.upper() == "POST":
                        # Use retry wrapper for throttling protection
                        response = self._execute_with_retry(
                            lambda url=url, request_params=request_params: requests.post(
                                url,
                                headers=self._get_headers(),
                                json=request_params,
                                timeout=120,
                            )
                        )
                    else:
                        if "maxResults" not in request_params:
                            request_params["maxResults"] = self.config.max_results
                        # Use retry wrapper for throttling protection
                        response = self._execute_with_retry(
                            lambda url=url, request_params=request_params: requests.request(
                                method=method.upper(),
                                url=url,
                                headers=self._get_headers(),
                                params=request_params,
                                timeout=120,
                            )
                        )
                    response.raise_for_status()
                    data = response.json()
                except requests.RequestException as e:
                    logger.error(f"API call failed: {str(e)}")
                    if e.response is not None:
                        logger.error(f"Response content: {e.response.text}")
                    return results if results else None

                if not isinstance(data, dict) or data_key not in data:
                    raise ValueError(f"Unexpected response format: {data}")

                results.extend(data[data_key])
                continuation_token = data.get("continuationToken")
                if not continuation_token:
                    break
            return results

    async def get_workspaces(self) -> List[Dict]:
        """Get all available workspaces"""
        return await self._make_request("workspaces", use_pagination=True)

    async def get_workspace(self, workspace_id: str) -> Dict:
        """Get a specific workspace by ID

        Args:
            workspace_id: ID of the workspace

        Returns:
            A dictionary containing the workspace details or an error message.
        """
        return await self._make_request(f"workspaces/{workspace_id}")

    async def get_lakehouses(self, workspace_id: str) -> List[Dict]:
        """Get all lakehouses in a workspace"""
        return await self.get_items(workspace_id=workspace_id, item_type="Lakehouse")

    async def get_warehouses(self, workspace_id: str) -> List[Dict]:
        """Get all warehouses in a workspace
        Args:
            workspace_id: ID of the workspace
        Returns:
            A list of dictionaries containing warehouse details or an error message.
        """
        return await self.get_items(workspace_id=workspace_id, item_type="Warehouse")

    async def get_tables(self, workspace_id: str, rsc_id: str, type: str) -> List[Dict]:
        """Get all tables in a lakehouse
        Args:
            workspace_id: ID of the workspace
            rsc_id: ID of the lakehouse
            type: Type of the resource (e.g., "Lakehouse" or "Warehouse")
        Returns:
            A list of dictionaries containing table details or an error message.
        """
        return await self._make_request(
            f"workspaces/{workspace_id}/{type}s/{rsc_id}/tables",
            use_pagination=True,
            data_key="data",
        )

    async def get_reports(self, workspace_id: str) -> List[Dict]:
        """Get all reports in a lakehouse
        Args:
            workspace_id: ID of the workspace
        Returns:
            A list of dictionaries containing report details or an error message.
        """
        return await self._make_request(
            f"workspaces/{workspace_id}/reports",
            use_pagination=True,
            data_key="value",
        )

    async def get_report(self, workspace_id: str, report_id: str) -> Dict:
        """Get a specific report by ID

        Args:
            workspace_id: ID of the workspace
            report_id: ID of the report

        Returns:
            A dictionary containing the report details or an error message.
        """
        return await self._make_request(
            f"workspaces/{workspace_id}/reports/{report_id}"
        )

    async def get_semantic_models(self, workspace_id: str) -> List[Dict]:
        """Get all semantic models in a lakehouse"""
        return await self._make_request(
            f"workspaces/{workspace_id}/semanticModels",
            use_pagination=True,
            data_key="value",
        )

    async def get_semantic_model(self, workspace_id: str, model_id: str) -> Dict:
        """Get a specific semantic model by ID"""
        return await self._make_request(
            f"workspaces/{workspace_id}/semanticModels/{model_id}"
        )

    async def resolve_workspace(self, workspace: str) -> str:
        """Convert workspace name or ID to workspace ID with caching"""
        return await self._cached_resolve_workspace(workspace)

    async def _resolve_workspace(self, workspace: str) -> str:
        """Internal method to convert workspace name or ID to workspace ID"""
        if _is_valid_uuid(workspace):
            return workspace

        workspaces = await self.get_workspaces()
        matching_workspaces = [
            w for w in workspaces if w["displayName"].lower() == workspace.lower()
        ]

        if not matching_workspaces:
            raise ValueError(f"No workspaces found with name: {workspace}")
        if len(matching_workspaces) > 1:
            raise ValueError(f"Multiple workspaces found with name: {workspace}")

        return matching_workspaces[0]["id"]

    async def resolve_lakehouse(self, workspace_id: str, lakehouse: str) -> str:
        """Convert lakehouse name or ID to lakehouse ID with caching"""
        return await self._cached_resolve_lakehouse(workspace_id, lakehouse)

    async def _resolve_lakehouse(self, workspace_id: str, lakehouse: str) -> str:
        """Internal method to convert lakehouse name or ID to lakehouse ID"""
        if _is_valid_uuid(lakehouse):
            return lakehouse

        lakehouses = await self.get_lakehouses(workspace_id)
        matching_lakehouses = [
            lh for lh in lakehouses if lh["displayName"].lower() == lakehouse.lower()
        ]

        if not matching_lakehouses:
            raise ValueError(f"No lakehouse found with name: {lakehouse}")
        if len(matching_lakehouses) > 1:
            raise ValueError(f"Multiple lakehouses found with name: {lakehouse}")

        return matching_lakehouses[0]["id"]

    async def get_items(
        self,
        workspace_id: str,
        item_type: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Get all items in a workspace"""
        if not _is_valid_uuid(workspace_id):
            raise ValueError("Invalid workspace ID.")
        if item_type:
            params = params or {}
            params["type"] = item_type
        return await self._make_request(
            f"workspaces/{workspace_id}/items", params=params, use_pagination=True
        )

    async def get_item(
        self,
        item_id: str,
        workspace_id: str,
        item_type: Optional[str] = None,
    ) -> Dict:
        """Get a specific item by ID"""

        if not _is_valid_uuid(item_id):
            item_name, item_id = await self.resolve_item_name_and_id(item_id)
        if not _is_valid_uuid(workspace_id):
            (workspace_name, workspace_id) = await self.resolve_workspace_name_and_id(
                workspace_id
            )
        return await self._make_request(
            f"workspaces/{workspace_id}/{item_type}s/{item_id}"
        )

    async def create_item(
        self,
        name: str,
        type: str,
        description: Optional[str] = None,
        definition: Optional[dict] = None,
        workspace: Optional[str | UUID] = None,
        lro: Optional[bool] = False,
    ):
        """
        Creates an item in a Fabric workspace.

        Parameters
        ----------
        name : str
            The name of the item to be created.
        type : str
            The type of the item to be created.
        description : str, default=None
            A description of the item to be created.
        definition : dict, default=None
            The definition of the item to be created.
        workspace : str | uuid.UUID, default=None
            The Fabric workspace name or ID.
            Defaults to None which resolves to the workspace of the attached lakehouse
            or if no lakehouse attached, resolves to the workspace of the notebook.
        """
        from sempy_labs._utils import item_types

        if _is_valid_uuid(workspace):
            workspace_id = workspace
        else:
            (workspace_name, workspace_id) = await self.resolve_workspace_name_and_id(
                workspace
            )
        item_type = item_types.get(type)[0].lower()

        payload = {
            "displayName": name,
        }
        if description:
            payload["description"] = description
        if definition:
            payload["definition"] = definition

        try:
            response = await self._make_request(
                endpoint=f"workspaces/{workspace_id}/{item_type}s",
                method="post",
                params=payload,
                lro=lro,
                lro_poll_interval=0.5,
            )
        except requests.RequestException as e:
            logger.error(f"API call failed: {str(e)}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise ValueError(
                f"Failed to create item '{name}' of type '{item_type}' in the '{workspace_id}' workspace."
            )        
        
        # Check if response contains an error
        if isinstance(response, dict):
            if "error" in response:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                logger.error(f"API error creating item: {error_msg}")
                raise ValueError(f"Failed to create item '{name}': {error_msg}")
            
            # Check if item was created successfully
            if "id" in response:
                logger.info(f"Successfully created item '{name}' with ID: {response['id']}")
                return response
            
            # If no ID and no error, log the full response for debugging
            logger.warning(f"Unexpected response format: {response}")
        
        # Legacy check - may not be reliable for all item types
        if hasattr(response, 'get') and response.get("displayName") and response.get("displayName") != name:
            logger.warning(f"Response displayName '{response.get('displayName')}' doesn't match requested name '{name}', but this may be normal")
        
        return response

    async def resolve_item_name_and_id(
        self,
        item: str | UUID,
        type: Optional[str] = None,
        workspace: Optional[str | UUID] = None,
    ) -> Tuple[str, UUID]:
        (workspace_name, workspace_id) = await self.resolve_workspace_name_and_id(
            workspace
        )
        item_id = await self.resolve_item_id(
            item=item, type=type, workspace=workspace_id
        )
        item_data = await self._make_request(
            f"workspaces/{workspace_id}/items/{item_id}"
        )
        item_name = item_data.get("displayName")
        return item_name, item_id

    async def resolve_item_id(
        self,
        item: str | UUID,
        type: Optional[str] = None,
        workspace: Optional[str | UUID] = None,
    ) -> UUID:
        (workspace_name, workspace_id) = await self.resolve_workspace_name_and_id(
            workspace
        )
        item_id = None

        if _is_valid_uuid(item):
            # Check (optional)
            item_id = item
            try:
                self._make_request(
                    endpoint=f"workspaces/{workspace_id}/items/{item_id}"
                )
            except requests.RequestException:
                raise ValueError(
                    f"The '{item_id}' item was not found in the '{workspace_name}' workspace."
                )
        else:
            if type is None:
                raise ValueError(
                    "The 'type' parameter is required if specifying an item name."
                )
            responses = await self._make_request(
                endpoint=f"workspaces/{workspace_id}/items?type={type}",
                use_pagination=True,
            )
            for v in responses:
                display_name = v["displayName"]
                if display_name == item:
                    item_id = v.get("id")
                    break

        if item_id is None:
            raise ValueError(
                f"There's no item '{item}' of type '{type}' in the '{workspace_name}' workspace."
            )

        return item_id

    async def resolve_workspace_name_and_id(
        self,
        workspace: Optional[str | UUID] = None,
    ) -> Tuple[str, UUID]:
        """
        Obtains the name and ID of the Fabric workspace.

        Parameters
        ----------
        workspace : str | uuid.UUID, default=None
            The Fabric workspace name or ID.
            Defaults to None which resolves to the workspace of the attached lakehouse
            or if no lakehouse attached, resolves to the workspace of the notebook.

        Returns
        -------
        str, uuid.UUID
            The name and ID of the Fabric workspace.
        """
        logger.debug(f"Resolving workspace name and ID for: {workspace}")
        if workspace is None:
            raise ValueError("Workspace must be specified.")
        elif _is_valid_uuid(workspace):
            workspace_id = workspace
            workspace_name = await self.resolve_workspace_name(workspace_id)
            return workspace_name, workspace_id
        else:
            responses = await self._make_request(
                endpoint="workspaces", use_pagination=True
            )
            workspace_id = None
            workspace_name = None
            for r in responses:
                display_name = r.get("displayName")
                if display_name == workspace:
                    workspace_name = workspace
                    workspace_id = r.get("id")
                    return workspace_name, workspace_id

        if workspace_name is None or workspace_id is None:
            raise ValueError("Workspace not found")

        return workspace_name, workspace_id

    async def resolve_workspace_name(self, workspace_id: Optional[UUID] = None) -> str:
        try:
            response = await self._make_request(endpoint=f"workspaces/{workspace_id}")
            if not response or "displayName" not in response:
                raise ValueError(
                    f"Workspace '{workspace_id}' not found or API response invalid: {response}"
                )
        except requests.RequestException:
            raise ValueError(f"The '{workspace_id}' workspace was not found.")

        return response.get("displayName")

    async def get_notebooks(self, workspace_id: str) -> List[Dict]:
        """Get all notebooks in a workspace"""
        return await self.get_items(workspace_id=workspace_id, item_type="Notebook")

    async def get_notebook(self, workspace_id: str, notebook_id: str) -> Dict:
        """Get a specific notebook by ID"""
        return await self.get_item(
            item_id=notebook_id, workspace_id=workspace_id, item_type="notebook"
        )

    async def get_notebook_definition(self, workspace_id: str, notebook_id: str) -> Dict:
        """Get the definition (content) of a specific notebook.

        Uses the getDefinition endpoint which returns the actual notebook content
        including the .ipynb payload.

        Args:
            workspace_id: ID of the workspace
            notebook_id: ID of the notebook

        Returns:
            Dictionary containing the notebook definition with parts including the ipynb content.
        """
        if not _is_valid_uuid(workspace_id):
            raise ValueError("Invalid workspace ID.")
        if not _is_valid_uuid(notebook_id):
            raise ValueError("Invalid notebook ID.")

        return await self._make_request(
            f"workspaces/{workspace_id}/notebooks/{notebook_id}/getDefinition",
            method="POST",
            lro=True,
            lro_poll_interval=1,
            lro_timeout=120
        )

    async def create_notebook(
        self, workspace_id: str, notebook_name: str, ipynb_name: str, content: str
    ) -> Dict:
        """Create a new notebook."""
        if not _is_valid_uuid(workspace_id):
            raise ValueError("Invalid workspace ID.")

        # Define the notebook definition
        logger.debug(
            f"Defining notebook '{notebook_name}' in workspace '{workspace_id}'."
        )
        definition = {
            "format": "ipynb",
            "parts": [
                {
                    "path": f"{ipynb_name}.ipynb",
                    "payload": base64.b64encode(
                        content
                        if isinstance(content, bytes)
                        else content.encode("utf-8")
                    ).decode("utf-8"),
                    "payloadType": "InlineBase64",
                },
                # {
                #     "path": ".platform",
                #     "payload": base64.b64encode("dotPlatformBase64String".encode("utf-8")).decode("utf-8"),
                #     "payloadType": "InlineBase64",
                # },
            ],
        }
        logger.info(
            f"-------Creating notebook '{notebook_name}' in workspace '{workspace_id}'."
        )
        return await self.create_item(
            workspace=workspace_id,
            type="Notebook",
            name=notebook_name,
            definition=definition,
        )

    async def update_notebook_definition(
        self, workspace_id: str, notebook_id: str, content: str, ipynb_name: str
    ) -> Dict:
        """Update the definition (content) of an existing notebook.

        Uses the updateDefinition endpoint to update the notebook content.

        Args:
            workspace_id: ID of the workspace
            notebook_id: ID of the notebook
            content: The notebook content as JSON string
            ipynb_name: The name of the ipynb file (without extension)

        Returns:
            Dictionary with the update result or error.
        """
        if not _is_valid_uuid(workspace_id):
            raise ValueError("Invalid workspace ID.")
        if not _is_valid_uuid(notebook_id):
            raise ValueError("Invalid notebook ID.")

        # Build the definition payload
        definition = {
            "definition": {
                "format": "ipynb",
                "parts": [
                    {
                        "path": f"{ipynb_name}.ipynb",
                        "payload": base64.b64encode(
                            content
                            if isinstance(content, bytes)
                            else content.encode("utf-8")
                        ).decode("utf-8"),
                        "payloadType": "InlineBase64",
                    }
                ],
            }
        }

        logger.info(f"Updating notebook '{notebook_id}' in workspace '{workspace_id}'.")

        return await self._make_request(
            f"workspaces/{workspace_id}/notebooks/{notebook_id}/updateDefinition",
            method="POST",
            params=definition,
            lro=True,
            lro_poll_interval=1,
            lro_timeout=120
        )
