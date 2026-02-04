# Fix: Enhanced get_job_details Implementation

## Problem Statement
The `get_job_details` function was incomplete and only returned basic timing information and a submitter GUID. It was missing critical information that users need when investigating job failures:

- **Notebook name and ID** that ran the job
- **Error messages and stack traces** for failed jobs
- **Input parameters** passed to the job
- **Spark application ID** and logs URL
- **Links to monitoring** tools (DAG view, Spark UI)
- **Resource configuration** details

## Solution Overview
Enhanced the `get_job_details` function to provide comprehensive job execution information by:

1. **Leveraging LivySession schema fields** that were previously ignored
2. **Adding Job Instance API integration** to retrieve detailed error information
3. **Including monitoring URLs** for Spark UI and Fabric portal
4. **Displaying resource configuration** (driver/executor settings)
5. **Improving error reporting** with detailed failure reasons

## Changes Made

### 1. Enhanced SparkClient (`helpers/clients/spark_client.py`)

Added new method to retrieve job instance details:

```python
async def get_job_instance(
    self, workspace_id: str, item_id: str, job_instance_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a job instance (execution details).

    Returns:
        Job instance details including failure information or None if not found
    """
```

This method calls the Fabric Job Instance API: `/workspaces/{workspaceId}/items/{itemId}/jobs/instances/{jobInstanceId}`

### 2. Enhanced get_job_details Function (`tools/notebook.py`)

The function now returns comprehensive information organized into the following sections:

#### Basic Information
- Job ID (Livy ID)
- **Notebook Name** (from `itemName` field)
- **Notebook ID** (from `item.id` field)
- **Job Instance ID** (for linking to DAG/Run)
- Job Type and Operation Name

#### Status Information
- Current state with emoji indicators
- Cancellation reason (if cancelled)

#### Timing Information
- Submitted, Started, Ended timestamps
- **Queued Duration** (from API)
- **Running Duration** (from API)
- **Total Duration** (from API or calculated)

#### Submitter Information
- Submitter ID and Type

#### Resource Configuration
- **Driver resources** (cores and memory)
- **Executor resources** (count, cores, memory)
- **Dynamic allocation settings** (if enabled)
- **Runtime version**
- **High concurrency mode** status

#### Spark Application Details
- **Spark Application ID** (for Spark UI access)
- **Spark UI URL** (clickable link)

#### Error Information (for Failed jobs)
- **Failure Reason** from Job Instance API
- **Root Activity ID** for debugging
- **Detailed error messages** with error codes and sources
- Guidance when no specific errors are available

#### Additional Metadata
- Capacity ID
- Workspace ID
- Origin
- Attempt Number

#### Monitoring & Logs
- Job Instance ID
- Spark Application UI link
- Workspace link
- Notebook link

## API Schema References

### LivySession Schema Fields Used
Based on `resources/openapi-specs/common/spark_definitions.json`:

```json
{
  "livyId": "UUID",
  "itemName": "string",           // ✅ Now captured
  "jobInstanceId": "UUID",         // ✅ Now captured
  "sparkApplicationId": "string",  // ✅ Now captured
  "state": "LivySessionState",
  "submittedDateTime": "string",
  "startDateTime": "string",
  "endDateTime": "string",
  "queuedDuration": "Duration",    // ✅ Now displayed
  "runningDuration": "Duration",   // ✅ Now displayed
  "totalDuration": "Duration",     // ✅ Now displayed
  "driverMemory": "integer",       // ✅ Now displayed
  "driverCores": "integer",        // ✅ Now displayed
  "executorMemory": "integer",     // ✅ Now displayed
  "executorCores": "integer",      // ✅ Now displayed
  "numExecutors": "integer",       // ✅ Now displayed
  "isDynamicAllocationEnabled": "boolean",  // ✅ Now displayed
  "runtimeVersion": "string",      // ✅ Now displayed
  "isHighConcurrency": "boolean",  // ✅ Now displayed
  "cancellationReason": "string",  // ✅ Now displayed
  "operationName": "string"        // ✅ Now displayed
}
```

### Job Instance API
Endpoint: `/workspaces/{workspaceId}/items/{itemId}/jobs/instances/{jobInstanceId}`

Returns:
```json
{
  "id": "UUID",
  "itemId": "UUID",
  "jobType": "string",
  "status": "string",
  "rootActivityId": "UUID",        // ✅ Now captured for debugging
  "failureReason": "string"        // ✅ Now displayed for failed jobs
}
```

## Example Output

### Before (Incomplete)
```markdown
# Spark Job Details

**Job ID:** 9d933543-cd67-483c-8218-4d6fee99607f
**Workspace:** MyWorkspace
**Notebook:** Run DAG

## Status
**State:** ❌ Failed

## Timing
**Submitted:** 2024-01-15T10:30:00Z
**Started:** 2024-01-15T10:31:00Z
**Ended:** 2024-01-15T10:35:00Z

## Submitter
**ID:** 12345678-1234-1234-1234-123456789abc
**Type:** User
```

### After (Complete)
```markdown
# Spark Job Details

## Basic Information
**Job ID (Livy ID):** 9d933543-cd67-483c-8218-4d6fee99607f
**Workspace:** MyWorkspace
**Notebook Name:** Run DAG
**Notebook ID:** abcd1234-5678-90ab-cdef-1234567890ab
**Job Instance ID:** xyz789-4567-89ab-cdef-1234567890ab
**Job Type:** SparkBatch
**Operation:** Notebook run

## Status
**State:** ❌ Failed

## Timing
**Submitted:** 2024-01-15T10:30:00Z
**Started:** 2024-01-15T10:31:00Z
**Ended:** 2024-01-15T10:35:00Z
**Queued Duration:** PT1M
**Running Duration:** PT4M
**Total Duration:** PT5M

## Submitter
**ID:** 12345678-1234-1234-1234-123456789abc
**Type:** User

## Resource Configuration
**Driver:** 4 cores, 28 GB memory
**Executors:** 2 executors × 4 cores × 28 GB memory
**Runtime Version:** 1.2
**High Concurrency Mode:** Disabled

## Spark Application
**Spark Application ID:** application_1234567890_0001
**Spark UI URL:** [View Spark UI](https://spark.fabric.microsoft.com/sparkui/application_1234567890_0001)

## Error Information
**Failure Reason:** Job failed due to stage failure: Task failed with exception

**Root Activity ID:** 8c2ee553-53a4-7edb-1042-0d8189a9e0ca

**Error Details:**
- **SparkException** (Source: Stage 0)
  org.apache.spark.SparkException: Task failed with exception
  at org.apache.spark.scheduler.TaskSetManager.handleFailedTask
  ...

## Additional Information
**Capacity ID:** capacity-123-456
**Workspace ID:** workspace-789-012
**Origin:** SubmittedJob
**Attempt Number:** 1 of 3

## Monitoring & Logs
- **Job Instance ID:** `xyz789-4567-89ab-cdef-1234567890ab`
- **Spark Application UI:** [Open UI](https://spark.fabric.microsoft.com/sparkui/application_1234567890_0001)
- **Workspace:** [MyWorkspace](https://app.fabric.microsoft.com/groups/workspace-789-012)
- **Notebook:** [Run DAG](https://app.fabric.microsoft.com/groups/workspace-789-012/notebooks/abcd1234-5678-90ab-cdef-1234567890ab)
```

## Benefits

1. **Complete Context**: Users now see which notebook ran and can navigate directly to it
2. **Actionable Error Information**: Detailed error messages and stack traces help debug failures
3. **Resource Visibility**: Clear view of compute resources allocated to the job
4. **Quick Access to Logs**: Direct links to Spark UI and Fabric portal
5. **Better Debugging**: Root Activity ID and Job Instance ID for correlation with logs
6. **Performance Insights**: Queued, running, and total duration help identify bottlenecks

## Testing Recommendations

1. Test with a **successful job** to verify all fields display correctly
2. Test with a **failed job** to ensure error information is captured
3. Test with a **cancelled job** to verify cancellation reason is shown
4. Test with **different notebook names** to ensure name resolution works
5. Test with **high concurrency mode** enabled
6. Test with **dynamic allocation** enabled

## Future Enhancements

1. **Input Parameters**: Add support for displaying notebook parameters passed to the job
2. **Output Artifacts**: Link to output files/tables created by the job
3. **Performance Metrics**: Include Spark metrics (shuffle read/write, task duration, etc.)
4. **Job Lineage**: Show parent/child job relationships for pipelines
5. **Historical Comparison**: Compare current run with previous executions
