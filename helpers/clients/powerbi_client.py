"""
Power BI REST API Client for DAX queries and report operations.

This client is separate from the Fabric API client because Power BI REST API
uses a different scope (analysis.windows.net) and base URL (api.powerbi.com).

Performance optimizations:
- Shared httpx.AsyncClient with connection pooling (TCP/TLS reuse)
- Application-level token caching with early refresh
- Non-blocking async retry with exponential backoff
"""

from typing import Dict, Any, List, Optional
import httpx
import asyncio
import time
from helpers.logging_config import get_logger
from helpers.utils import _is_valid_uuid

logger = get_logger(__name__)

# Power BI API scope (different from Fabric API scope)
POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"

# Module-level shared HTTP client for connection pooling
_shared_client: Optional[httpx.AsyncClient] = None


def _get_shared_client() -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
    return _shared_client


class PowerBIClient:
    """Client for Power BI REST API (separate from Fabric API).

    Used for DAX query execution and report operations that require
    the Power BI API scope instead of the Fabric API scope.
    """

    BASE_URL = "https://api.powerbi.com/v1.0/myorg"

    def __init__(self, credential, config: Optional[Dict] = None):
        """Initialize the Power BI client.

        Args:
            credential: Azure credential for authentication
            config: Optional configuration dictionary with:
                - max_retries: Max retry attempts for throttled requests (default: 3)
                - default_retry_after: Default wait time in seconds (default: 60)
                - enable_exponential_backoff: Whether to use exponential backoff (default: True)
        """
        self.credential = credential
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 3)
        self.default_retry_after = self.config.get("default_retry_after", 60)
        self.enable_exponential_backoff = self.config.get("enable_exponential_backoff", True)
        self._client = _get_shared_client()
        # Token caching - avoids redundant get_token() calls
        self._token: Optional[str] = None
        self._token_expiry: float = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with cached auth token (refreshed 5 min before expiry)."""
        now = time.time()
        if not self._token or (self._token_expiry - now) < 300:
            access_token = self.credential.get_token(POWERBI_SCOPE)
            self._token = access_token.token
            self._token_expiry = access_token.expires_on
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }

    def _build_url(self, endpoint: str, workspace_id: Optional[str] = None) -> str:
        """Build the full URL for an API endpoint.

        Args:
            endpoint: The API endpoint path
            workspace_id: Optional workspace ID for workspace-scoped endpoints

        Returns:
            The full URL
        """
        if workspace_id:
            return f"{self.BASE_URL}/groups/{workspace_id}/{endpoint}"
        return f"{self.BASE_URL}/{endpoint}"

    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> httpx.Response:
        """Execute an HTTP request with automatic retry on throttling (429 responses).
        Auth headers are refreshed on each retry attempt.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The full URL to request
            max_retries: Maximum number of retries (defaults to config value)
            **kwargs: Additional arguments passed to httpx.AsyncClient.request()

        Returns:
            The successful httpx.Response

        Raises:
            httpx.HTTPStatusError: If all retries are exhausted
        """
        max_retries = max_retries or self.max_retries
        retry_count = 0

        while retry_count <= max_retries:
            response = await self._client.request(
                method, url, headers=self._get_headers(), **kwargs
            )

            if response.status_code == 429:  # Too Many Requests (Throttled)
                if retry_count >= max_retries:
                    logger.error(f"Request failed after {max_retries} retries due to throttling")
                    response.raise_for_status()

                # Get Retry-After header (defaults to configured value)
                retry_after = int(response.headers.get("Retry-After", self.default_retry_after))

                # Apply exponential backoff if enabled
                if self.enable_exponential_backoff and retry_count > 0:
                    backoff_multiplier = min(2 ** retry_count, 4)
                    retry_after = min(retry_after * backoff_multiplier, retry_after * 2)

                logger.warning(
                    f"Power BI API throttled (429). Waiting {retry_after}s before retry "
                    f"{retry_count + 1}/{max_retries}."
                )

                await asyncio.sleep(retry_after)
                retry_count += 1
                continue

            return response

        return response

    async def execute_dax(
        self,
        workspace_id: str,
        dataset_id: str,
        dax_query: str,
        include_nulls: bool = True,
        impersonated_user_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a DAX query against a semantic model.

        Args:
            workspace_id: The workspace ID (GUID)
            dataset_id: The dataset/semantic model ID (GUID)
            dax_query: The DAX query to execute (e.g., "EVALUATE 'TableName'")
            include_nulls: Whether to include null values in results (default: True)
            impersonated_user_name: Optional user to impersonate for RLS testing

        Returns:
            Dictionary with query results containing:
            - results: List of result tables
            - Each table has 'rows' (data) and 'columns' (schema)

        Raises:
            ValueError: If the query fails or returns an error
        """
        url = self._build_url(f"datasets/{dataset_id}/executeQueries", workspace_id)

        payload = {
            "queries": [{"query": dax_query}],
            "serializerSettings": {"includeNulls": include_nulls}
        }

        if impersonated_user_name:
            payload["impersonatedUserName"] = impersonated_user_name

        try:
            response = await self._execute_with_retry("POST", url, json=payload)

            if response.status_code == 400:
                try:
                    error_data = response.json()
                except Exception:
                    raise ValueError(f"DAX query error: HTTP 400 (empty response body)")
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                # Check for specific DAX errors
                if "error" in error_data:
                    details = error_data["error"].get("details", [])
                    if details:
                        error_message = details[0].get("message", error_message)
                raise ValueError(f"DAX query error: {error_message}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                except Exception:
                    pass
                else:
                    error_message = error_data.get("error", {}).get("message", str(e))
                    raise ValueError(f"DAX query failed: {error_message}")
            logger.error(f"DAX query execution failed: {e}")
            raise ValueError(f"DAX query execution failed: {str(e)}")

    async def get_report_pages(
        self,
        workspace_id: str,
        report_id: str
    ) -> List[Dict[str, Any]]:
        """Get all pages in a Power BI report.

        Args:
            workspace_id: The workspace ID (GUID)
            report_id: The report ID (GUID)

        Returns:
            List of page dictionaries with name, displayName, and order
        """
        url = self._build_url(f"reports/{report_id}/pages", workspace_id)

        try:
            response = await self._execute_with_retry("GET", url, timeout=60)

            response.raise_for_status()
            data = response.json()
            return data.get("value", [])

        except httpx.HTTPError as e:
            logger.error(f"Failed to get report pages: {e}")
            raise ValueError(f"Failed to get report pages: {str(e)}")

    async def export_report(
        self,
        workspace_id: str,
        report_id: str,
        format: str = "PDF",
        pages: Optional[List[Dict]] = None,
        bookmark_state: Optional[str] = None,
        report_level_filters: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Start an export job for a Power BI report.

        Args:
            workspace_id: The workspace ID (GUID)
            report_id: The report ID (GUID)
            format: Export format - "PDF", "PPTX", "PNG", "IMAGE", "XLSX", "DOCX", "CSV", "XML", "MHTML"
            pages: Optional list of pages to export with format:
                   [{"pageName": "ReportSection1", "visualName": "visual1"}]
            bookmark_state: Optional bookmark state to apply
            report_level_filters: Optional report-level filters

        Returns:
            Dictionary with export job info including 'id' for tracking

        Note:
            Requires Premium, Embedded, or Fabric capacity.
        """
        url = self._build_url(f"reports/{report_id}/ExportTo", workspace_id)

        payload = {
            "format": format
        }

        # Add optional parameters
        if pages:
            payload["powerBIReportConfiguration"] = {"pages": pages}
        if bookmark_state:
            if "powerBIReportConfiguration" not in payload:
                payload["powerBIReportConfiguration"] = {}
            payload["powerBIReportConfiguration"]["defaultBookmark"] = {
                "state": bookmark_state
            }
        if report_level_filters:
            if "powerBIReportConfiguration" not in payload:
                payload["powerBIReportConfiguration"] = {}
            payload["powerBIReportConfiguration"]["reportLevelFilters"] = report_level_filters

        try:
            response = await self._execute_with_retry("POST", url, json=payload)

            if response.status_code == 403:
                raise ValueError(
                    "Export failed: Requires Premium, Embedded, or Fabric capacity. "
                    "Export is not available for Pro-only workspaces."
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Failed to start report export: {e}")
            raise ValueError(f"Failed to start report export: {str(e)}")

    async def get_export_status(
        self,
        workspace_id: str,
        report_id: str,
        export_id: str
    ) -> Dict[str, Any]:
        """Check the status of an export job.

        Args:
            workspace_id: The workspace ID (GUID)
            report_id: The report ID (GUID)
            export_id: The export job ID returned from export_report

        Returns:
            Dictionary with status info:
            - status: "NotStarted", "Running", "Succeeded", "Failed"
            - percentComplete: Progress percentage
            - resourceLocation: Download URL when complete
        """
        url = self._build_url(f"reports/{report_id}/exports/{export_id}", workspace_id)

        try:
            response = await self._execute_with_retry("GET", url, timeout=60)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Failed to get export status: {e}")
            raise ValueError(f"Failed to get export status: {str(e)}")

    async def get_export_file(
        self,
        workspace_id: str,
        report_id: str,
        export_id: str
    ) -> bytes:
        """Download the exported report file.

        Args:
            workspace_id: The workspace ID (GUID)
            report_id: The report ID (GUID)
            export_id: The export job ID

        Returns:
            The file content as bytes
        """
        url = self._build_url(f"reports/{report_id}/exports/{export_id}/file", workspace_id)

        try:
            response = await self._execute_with_retry(
                "GET", url, timeout=300  # Longer timeout for file download
            )

            response.raise_for_status()
            return response.content

        except httpx.HTTPError as e:
            logger.error(f"Failed to download export file: {e}")
            raise ValueError(f"Failed to download export file: {str(e)}")

    async def wait_for_export(
        self,
        workspace_id: str,
        report_id: str,
        export_id: str,
        poll_interval: int = 5,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Wait for an export job to complete.

        Args:
            workspace_id: The workspace ID
            report_id: The report ID
            export_id: The export job ID
            poll_interval: Seconds between status checks (default: 5)
            timeout: Maximum seconds to wait (default: 300 / 5 minutes)

        Returns:
            Final export status with resourceLocation for download

        Raises:
            ValueError: If export fails or times out
        """
        start_time = time.time()

        while True:
            status = await self.get_export_status(workspace_id, report_id, export_id)
            current_status = status.get("status", "Unknown")

            if current_status == "Succeeded":
                logger.info("Export completed successfully")
                return status

            if current_status == "Failed":
                error = status.get("error", {}).get("message", "Unknown error")
                raise ValueError(f"Export failed: {error}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise ValueError(f"Export timed out after {timeout} seconds")

            percent = status.get("percentComplete", 0)
            logger.debug(f"Export status: {current_status}, {percent}% complete")

            await asyncio.sleep(poll_interval)

    async def resolve_dataset_id(
        self,
        workspace_id: str,
        dataset: str,
        fabric_client
    ) -> str:
        """Resolve a dataset name to its ID.

        Args:
            workspace_id: The workspace ID (GUID)
            dataset: The dataset name or ID
            fabric_client: FabricApiClient instance for resolving names

        Returns:
            The dataset ID (GUID)
        """
        if _is_valid_uuid(dataset):
            return dataset

        # Use Fabric API to list semantic models and find by name
        models = await fabric_client.get_semantic_models(workspace_id)

        if not models:
            raise ValueError(f"No semantic models found in workspace")

        matching = [m for m in models if m.get("displayName", "").lower() == dataset.lower()]

        if not matching:
            raise ValueError(f"No semantic model found with name: {dataset}")
        if len(matching) > 1:
            raise ValueError(f"Multiple semantic models found with name: {dataset}")

        return matching[0]["id"]

    async def resolve_report_id(
        self,
        workspace_id: str,
        report: str,
        fabric_client
    ) -> str:
        """Resolve a report name to its ID.

        Args:
            workspace_id: The workspace ID (GUID)
            report: The report name or ID
            fabric_client: FabricApiClient instance for resolving names

        Returns:
            The report ID (GUID)
        """
        if _is_valid_uuid(report):
            return report

        # Use Fabric API to list reports and find by name
        reports = await fabric_client.get_reports(workspace_id)

        if not reports:
            raise ValueError(f"No reports found in workspace")

        matching = [r for r in reports if r.get("displayName", "").lower() == report.lower()]

        if not matching:
            raise ValueError(f"No report found with name: {report}")
        if len(matching) > 1:
            raise ValueError(f"Multiple reports found with name: {report}")

        return matching[0]["id"]

    async def get_refresh_history(
        self,
        workspace_id: str,
        dataset_id: str,
        top: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the refresh history for a semantic model.

        Args:
            workspace_id: The workspace ID (GUID)
            dataset_id: The dataset/semantic model ID (GUID)
            top: Number of refresh entries to return (default: 10, max: 100)

        Returns:
            List of refresh history entries with:
            - id: Refresh request ID
            - refreshType: Type of refresh (Scheduled, OnDemand, ViaApi, etc.)
            - startTime: When the refresh started (ISO 8601)
            - endTime: When the refresh ended (ISO 8601)
            - status: Completion status (Completed, Failed, Unknown, Disabled, Cancelled)
            - serviceExceptionJson: Error details if failed
        """
        top = min(max(1, top), 100)  # Clamp between 1 and 100
        url = self._build_url(f"datasets/{dataset_id}/refreshes?$top={top}", workspace_id)

        try:
            response = await self._execute_with_retry("GET", url, timeout=60)

            response.raise_for_status()
            data = response.json()
            return data.get("value", [])

        except httpx.HTTPError as e:
            logger.error(f"Failed to get refresh history: {e}")
            raise ValueError(f"Failed to get refresh history: {str(e)}")
