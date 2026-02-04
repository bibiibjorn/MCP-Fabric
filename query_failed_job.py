"""
Query failed Fabric job details for "Run DAG (FTX-ACC-ETL)"
Searches across workspaces to find the failed job and retrieve error details.
"""

import asyncio
from helpers.clients.fabric_client import FabricApiClient
from helpers.clients.spark_client import SparkClient
from helpers.utils.authentication import get_shared_credential


async def find_failed_dag_job():
    """Find and display details about the failed DAG job."""

    # Initialize clients with shared credential (uses cached auth or opens browser)
    credential = get_shared_credential()
    fabric_client = FabricApiClient(credential=credential)
    spark_client = SparkClient(fabric_client)

    print("=" * 70)
    print("Searching for failed 'Run DAG (FTX-ACC-ETL)' job...")
    print("=" * 70)

    # Get all workspaces
    print("\n[1] Fetching workspaces...")
    workspaces = await fabric_client.get_workspaces()
    print(f"    Found {len(workspaces)} workspaces")

    # Search for failed jobs in each workspace
    all_failed_jobs = []

    print("\n[2] Scanning workspaces for failed jobs...")
    for ws in workspaces:
        ws_id = ws.get('id')
        ws_name = ws.get('displayName', 'Unknown')

        try:
            # Get Livy sessions (Spark jobs) for this workspace
            sessions = await spark_client.list_workspace_sessions(
                ws_id,
                filters={"state": "Failed"},
                max_results=20
            )

            for session in sessions:
                item_name = session.get('itemName', '')
                # Look for "Run DAG" or "FTX-ACC-ETL" in the name
                if 'Run DAG' in item_name or 'FTX' in item_name or 'ETL' in item_name:
                    session['_workspace_name'] = ws_name
                    session['_workspace_id'] = ws_id
                    all_failed_jobs.append(session)
                    print(f"    Found: {item_name} in {ws_name}")

        except Exception as e:
            # Skip workspaces we can't access
            pass

    if not all_failed_jobs:
        print("\n    No matching failed jobs found.")
        print("\n    Listing ALL recent failed jobs across workspaces...")

        # Show all failed jobs instead
        for ws in workspaces:
            ws_id = ws.get('id')
            ws_name = ws.get('displayName', 'Unknown')

            try:
                sessions = await spark_client.list_workspace_sessions(
                    ws_id,
                    filters={"state": "Failed"},
                    max_results=5
                )

                for session in sessions:
                    session['_workspace_name'] = ws_name
                    session['_workspace_id'] = ws_id
                    all_failed_jobs.append(session)

            except Exception:
                pass

    # Sort by submitted time
    all_failed_jobs.sort(key=lambda x: x.get('submittedDateTime', ''), reverse=True)

    print(f"\n[3] Found {len(all_failed_jobs)} failed jobs total")

    # Display the failed jobs
    print("\n" + "=" * 70)
    print("FAILED JOBS")
    print("=" * 70)

    for i, job in enumerate(all_failed_jobs[:10], 1):
        item_name = job.get('itemName', 'Unknown')
        ws_name = job.get('_workspace_name', 'Unknown')
        ws_id = job.get('_workspace_id', '')
        livy_id = job.get('livyId', 'N/A')
        job_instance_id = job.get('jobInstanceId', '')
        submitted = job.get('submittedDateTime', 'N/A')
        duration = job.get('totalDuration', 'N/A')
        item_info = job.get('item', {})
        item_id = item_info.get('id', '') if isinstance(item_info, dict) else ''

        print(f"\n--- Job {i} ---")
        print(f"  Notebook:      {item_name}")
        print(f"  Workspace:     {ws_name}")
        print(f"  Submitted:     {submitted}")
        print(f"  Duration:      {duration}")
        print(f"  Livy ID:       {livy_id}")
        print(f"  Job Instance:  {job_instance_id}")
        print(f"  Item ID:       {item_id}")

        # Try to get detailed error info
        if job_instance_id and item_id:
            print(f"\n  Fetching error details...")
            try:
                job_details = await spark_client.get_job_instance(ws_id, item_id, job_instance_id)
                if job_details:
                    failure_reason = (
                        job_details.get('failureReason') or
                        job_details.get('errorMessage') or
                        job_details.get('error') or
                        job_details.get('message')
                    )
                    if failure_reason:
                        print(f"  FAILURE REASON:")
                        print(f"  {failure_reason}")
                    else:
                        print(f"  No specific error captured (check Spark logs in Fabric portal)")

                    root_activity_id = job_details.get('rootActivityId')
                    if root_activity_id:
                        print(f"  Root Activity ID: {root_activity_id}")
            except Exception as e:
                print(f"  Could not fetch job details: {e}")

        # Provide portal links
        if ws_id and item_id:
            print(f"\n  Links:")
            print(f"  - Notebook: https://app.fabric.microsoft.com/groups/{ws_id}/notebooks/{item_id}")
            if job_instance_id:
                print(f"  - Monitoring: https://app.fabric.microsoft.com/groups/{ws_id}/monitoringhub?experience=fabric-developer&jobId={job_instance_id}")

    print("\n" + "=" * 70)
    print("TIP: For 'Job instance failed without detail error', check:")
    print("  1. Spark logs in the Fabric portal (Monitoring Hub)")
    print("  2. The notebook code for unhandled exceptions")
    print("  3. Downstream dependencies (lakehouses, data sources)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(find_failed_dag_job())
