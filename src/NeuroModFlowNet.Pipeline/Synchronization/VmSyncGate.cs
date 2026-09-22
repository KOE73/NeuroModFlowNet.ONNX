namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Ordered gate for one synchronization point in a VM program.
/// </summary>
/// <remarks>
/// Runs are released by dense accepted <c>RunId</c>. If run 12 fails before this gate, the host calls
/// <see cref="NotifyRunClosed"/> and the gate advances exactly as if run 12 had reached and passed this point.
/// </remarks>
public sealed class VmSyncGate
{
    readonly object syncRoot = new();
    readonly Dictionary<long, TaskCompletionSource> waiters = [];
    readonly HashSet<long> arrived = [];
    readonly HashSet<long> closedBeforeGate = [];
    readonly Dictionary<long, TaskCompletionSource> criticalWaiters = [];
    readonly HashSet<long> criticalWaiting = [];
    readonly HashSet<long> closedBeforeCriticalSection = [];
    long nextRunIdToRelease;
    long nextRunIdToEnterCriticalSection;
    long? criticalSectionOwnerRunId;

    public VmSyncGate(string name, long initialRunId = 0)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if(initialRunId < 0)
            throw new ArgumentOutOfRangeException(nameof(initialRunId), "Initial run id must be non-negative.");

        Name = name;
        nextRunIdToRelease = initialRunId;
        nextRunIdToEnterCriticalSection = initialRunId;
    }

    public string Name { get; }

    public ValueTask ArriveAndWaitAsync(long runId, CancellationToken cancellationToken)
    {
        if(runId < 0)
            throw new ArgumentOutOfRangeException(nameof(runId), "Run id must be non-negative.");

        Task waitTask;

        lock(syncRoot)
        {
            if(runId < nextRunIdToRelease)
                return ValueTask.CompletedTask;

            arrived.Add(runId);
            TaskCompletionSource waiter = GetWaiter(runId);
            TryReleaseReadyRuns();
            waitTask = waiter.Task;
        }

        return waitTask.IsCompletedSuccessfully
            ? ValueTask.CompletedTask
            : new ValueTask(waitTask.WaitAsync(cancellationToken));
    }

    public ValueTask AcquireInOrderAsync(long runId, CancellationToken cancellationToken)
    {
        if(runId < 0)
            throw new ArgumentOutOfRangeException(nameof(runId), "Run id must be non-negative.");

        Task waitTask;

        lock(syncRoot)
        {
            if(runId < nextRunIdToEnterCriticalSection)
                return ValueTask.CompletedTask;

            if(runId == nextRunIdToEnterCriticalSection && criticalSectionOwnerRunId is null)
            {
                criticalSectionOwnerRunId = runId;
                return ValueTask.CompletedTask;
            }

            criticalWaiting.Add(runId);
            TaskCompletionSource waiter = GetCriticalWaiter(runId);
            TryReleaseReadyCriticalSectionOwner();
            waitTask = waiter.Task;
        }

        return waitTask.IsCompletedSuccessfully
            ? ValueTask.CompletedTask
            : new ValueTask(waitTask.WaitAsync(cancellationToken));
    }

    public void Release(long runId)
    {
        if(runId < 0)
            throw new ArgumentOutOfRangeException(nameof(runId), "Run id must be non-negative.");

        lock(syncRoot)
        {
            if(runId < nextRunIdToEnterCriticalSection)
                return;

            if(criticalSectionOwnerRunId != runId)
                throw new InvalidOperationException($"Run {runId} cannot release critical section '{Name}' because it does not own it.");

            criticalSectionOwnerRunId = null;
            nextRunIdToEnterCriticalSection++;
            closedBeforeCriticalSection.Remove(runId);
            TryReleaseReadyCriticalSectionOwner();
        }
    }

    public void NotifyRunClosed(long runId)
    {
        if(runId < 0)
            return;

        lock(syncRoot)
        {
            if(runId < nextRunIdToRelease)
            {
                NotifyRunClosedBeforeCriticalSection(runId);
                return;
            }

            closedBeforeGate.Add(runId);
            TryReleaseReadyRuns();
            NotifyRunClosedBeforeCriticalSection(runId);
        }
    }

    TaskCompletionSource GetWaiter(long runId)
    {
        if(waiters.TryGetValue(runId, out TaskCompletionSource? waiter))
            return waiter;

        waiter = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        waiters.Add(runId, waiter);
        return waiter;
    }

    TaskCompletionSource GetCriticalWaiter(long runId)
    {
        if(criticalWaiters.TryGetValue(runId, out TaskCompletionSource? waiter))
            return waiter;

        waiter = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        criticalWaiters.Add(runId, waiter);
        return waiter;
    }

    void TryReleaseReadyRuns()
    {
        while(true)
        {
            if(arrived.Remove(nextRunIdToRelease))
            {
                if(waiters.Remove(nextRunIdToRelease, out TaskCompletionSource? waiter))
                    waiter.TrySetResult();

                closedBeforeGate.Remove(nextRunIdToRelease);
                nextRunIdToRelease++;
                continue;
            }

            if(closedBeforeGate.Remove(nextRunIdToRelease))
            {
                if(waiters.Remove(nextRunIdToRelease, out TaskCompletionSource? waiter))
                    waiter.TrySetException(new PrecedingRunFailedException(nextRunIdToRelease));

                nextRunIdToRelease++;
                continue;
            }

            break;
        }
    }

    void NotifyRunClosedBeforeCriticalSection(long runId)
    {
        if(runId < nextRunIdToEnterCriticalSection)
            return;

        if(criticalSectionOwnerRunId == runId)
        {
            criticalSectionOwnerRunId = null;
            nextRunIdToEnterCriticalSection++;
            TryReleaseReadyCriticalSectionOwner();
            return;
        }

        closedBeforeCriticalSection.Add(runId);
        TryReleaseReadyCriticalSectionOwner();
    }

    void TryReleaseReadyCriticalSectionOwner()
    {
        while(criticalSectionOwnerRunId is null)
        {
            if(closedBeforeCriticalSection.Remove(nextRunIdToEnterCriticalSection))
            {
                if(criticalWaiters.Remove(nextRunIdToEnterCriticalSection, out TaskCompletionSource? waiter))
                    waiter.TrySetException(new PrecedingRunFailedException(nextRunIdToEnterCriticalSection));

                criticalWaiting.Remove(nextRunIdToEnterCriticalSection);
                nextRunIdToEnterCriticalSection++;
                continue;
            }

            if(criticalWaiting.Remove(nextRunIdToEnterCriticalSection))
            {
                if(criticalWaiters.Remove(nextRunIdToEnterCriticalSection, out TaskCompletionSource? waiter))
                    waiter.TrySetResult();

                criticalSectionOwnerRunId = nextRunIdToEnterCriticalSection;
                return;
            }

            break;
        }
    }
}
