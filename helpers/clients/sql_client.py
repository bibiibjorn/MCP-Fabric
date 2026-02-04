import polars as pl
import mssql_python
import struct
from typing import Optional
from helpers.clients import FabricApiClient, LakehouseClient, WarehouseClient
from helpers.utils.authentication import get_shared_credential, SQL_SCOPE


# SQL_COPT_SS_ACCESS_TOKEN constant for token-based authentication
SQL_COPT_SS_ACCESS_TOKEN = 1256


def get_access_token_struct() -> bytes:
    """
    Get an access token for SQL authentication and pack it into the required format.

    Uses the shared credential from authentication.py to ensure the same
    cached auth is used for both REST API and SQL/TDS connections.

    Returns:
        bytes: The packed token structure for SQL_COPT_SS_ACCESS_TOKEN.
    """
    azure_credentials = get_shared_credential()
    token = azure_credentials.get_token(SQL_SCOPE).token
    # Encode token as UTF-16LE (Windows expects this format)
    token_bytes = token.encode("utf-16-le")
    # Pack as: 4-byte length (little-endian) + token bytes
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    return token_struct


def get_mssql_connection(server: str, database: str) -> mssql_python.Connection:
    """
    Creates a connection to SQL Server using mssql-python (no ODBC driver required).

    Uses token-based authentication with Microsoft Entra ID.

    Args:
        server (str): The server address.
        database (str): The database name.

    Returns:
        mssql_python.Connection: A database connection object.
    """
    # Connection string without auth params (token is passed via attrs_before)
    conn_str = f"Server={server};Database={database};Encrypt=Yes;TrustServerCertificate=No;"

    # Get the packed access token
    token_struct = get_access_token_struct()

    # Connect using the mssql-python driver with token authentication
    conn = mssql_python.connect(
        conn_str, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct}
    )
    return conn


async def get_sql_endpoint(
    workspace: str = None,
    lakehouse: Optional[str] = None,
    warehouse: Optional[str] = None,
    type: str = None,
) -> tuple:
    """
    Retrieve the SQL endpoint for a specified lakehouse or warehouse.

    Args:
        workspace: Name or ID of the workspace (optional).
        lakehouse: Name or ID of the lakehouse (optional).
        warehouse: Name or ID of the warehouse (optional).
        type: Type of resource ('lakehouse' or 'warehouse'). If not provided, inferred from lakehouse/warehouse params.

    Returns:
        A tuple (database, sql_endpoint) or (None, error_message) in case of error.
    """
    try:
        # Use shared credential for unified auth across REST API and SQL/TDS
        credential = get_shared_credential()
        fabClient = FabricApiClient(credential)

        # Infer type from parameters if not explicitly provided
        if type is None:
            if lakehouse:
                type = "lakehouse"
            elif warehouse:
                type = "warehouse"
            else:
                return None, "Either lakehouse or warehouse must be specified."

        resource_name = None
        endpoint = None

        # Resolve workspace name/ID
        workspace_name, workspace_id = await fabClient.resolve_workspace_name_and_id(
            workspace
        )

        if type.lower() == "lakehouse":
            if not lakehouse:
                return None, "Lakehouse name or ID must be specified for type 'lakehouse'."

            client = LakehouseClient(fabClient)
            resource_name, resource_id = await fabClient.resolve_item_name_and_id(
                workspace=workspace_id, item=lakehouse, type="Lakehouse"
            )

            # Use workspace_id (resolved) not workspace (original param)
            lakehouse_obj = await client.get_lakehouse(
                workspace=workspace_id, lakehouse=resource_id
            )

            endpoint = (
                lakehouse_obj.get("properties", {})
                .get("sqlEndpointProperties", {})
                .get("connectionString")
            )

        elif type.lower() == "warehouse":
            if not warehouse:
                return None, "Warehouse name or ID must be specified for type 'warehouse'."

            client = WarehouseClient(fabClient)
            resource_name, resource_id = await fabClient.resolve_item_name_and_id(
                workspace=workspace_id, item=warehouse, type="Warehouse"
            )

            # Use workspace_id (resolved) not workspace (original param)
            warehouse_obj = await client.get_warehouse(
                workspace=workspace_id, warehouse=resource_id
            )

            endpoint = warehouse_obj.get("properties", {}).get("connectionString")

        else:
            return None, f"Invalid type '{type}'. Must be 'lakehouse' or 'warehouse'."

        if resource_name and endpoint:
            return resource_name, endpoint
        else:
            return (
                None,
                f"No SQL endpoint found for {type} '{lakehouse or warehouse}' in workspace '{workspace_name}'.",
            )

    except Exception as e:
        return None, f"Error retrieving SQL endpoint: {str(e)}"


class SQLClient:
    """
    SQL Client for querying Fabric Lakehouses and Warehouses.

    Uses mssql-python driver which does NOT require ODBC driver installation.
    Authentication is handled via Microsoft Entra ID access tokens.
    """

    def __init__(self, sql_endpoint: str, database: str):
        self.server = sql_endpoint
        self.database = database
        self._conn = None

    def _get_connection(self) -> mssql_python.Connection:
        """Get or create a connection."""
        if self._conn is None:
            self._conn = get_mssql_connection(self.server, self.database)
        return self._conn

    def run_query(self, query: str) -> pl.DataFrame:
        """
        Execute a SQL query and return results as a Polars DataFrame.

        Args:
            query: The SQL query to execute.

        Returns:
            pl.DataFrame: Query results as a Polars DataFrame.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query)

        # Fetch column names from cursor description
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        # Fetch all rows
        rows = cursor.fetchall()

        # Convert to Polars DataFrame using dict-based approach
        # This handles Row objects and tuples correctly regardless of column count
        if rows:
            data = [dict(zip(columns, row)) for row in rows]
            return pl.DataFrame(data)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(schema={col: pl.Utf8 for col in columns})

    def close(self):
        """Close the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
