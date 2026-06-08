namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Live timing statistics for one VM slot in the multi-stream lab.
/// Thread-safe: updated by the VM worker task, read by the dashboard task.
/// </summary>
internal sealed class VmSlotStatistics
{
    readonly MovingAverage runTotalAverage = new(60);
    readonly MovingAverage instructionTotalAverage = new(60);
    readonly MovingAverage controllerTotalAverage = new(60);
    readonly Dictionary<string, MovingAverage> instructionAverages = new(StringComparer.Ordinal);
    readonly DateTimeOffset startedAt = DateTimeOffset.UtcNow;
    readonly object syncLock = new();

    public int VmIndex { get; }

    public VmSlotStatistics(int vmIndex)
    {
        VmIndex = vmIndex;
    }

    #region Snapshot properties (read by dashboard — values are double/int, safe on 64-bit without lock)

    public int CompletedFrames { get; private set; }

    public int FailedFrames { get; private set; }

    public TimeSpan Uptime => DateTimeOffset.UtcNow - startedAt;

    public double LastRunTotalMilliseconds { get; private set; }

    public double AverageRunTotalMilliseconds { get; private set; }

    public double LastInstructionTotalMilliseconds { get; private set; }

    public double AverageInstructionTotalMilliseconds { get; private set; }

    public double LastControllerTotalMilliseconds { get; private set; }

    public double AverageControllerTotalMilliseconds { get; private set; }

    public double ProcessingFps { get; private set; }

    #endregion

    #region Instruction timing snapshots

    public IReadOnlyList<InstructionTimingSnapshot> GetInstructionSnapshots()
    {
        lock(syncLock)
        {
            return instructionAverages
                .OrderBy(item => item.Key, StringComparer.Ordinal)
                .Select(item => new InstructionTimingSnapshot(item.Key, item.Value.Last, item.Value.Average))
                .ToArray();
        }
    }

    #endregion

    #region Mutation (called from VmWorker task)

    public void AddWarmup(VmRunOutcome outcome)
    {
        lock(syncLock)
            UpdateInstructionAveragesLocked(outcome.Trace);
    }

    public void AddFrame(VmRunOutcome outcome, TimeSpan runTotalTime)
    {
        lock(syncLock)
        {
            double runMs = runTotalTime.TotalMilliseconds;
            runTotalAverage.Add(runMs);
            LastRunTotalMilliseconds = runTotalAverage.Last;
            AverageRunTotalMilliseconds = runTotalAverage.Average;
            ProcessingFps = AverageRunTotalMilliseconds <= 0 ? 0 : 1000.0 / AverageRunTotalMilliseconds;

            if(outcome.Trace is { } trace)
            {
                instructionTotalAverage.Add(trace.TotalInstructionMilliseconds);
                LastInstructionTotalMilliseconds = instructionTotalAverage.Last;
                AverageInstructionTotalMilliseconds = instructionTotalAverage.Average;
                UpdateInstructionAveragesLocked(trace);
            }

            if(outcome.Timing is { } timing)
            {
                controllerTotalAverage.Add(timing.TotalMilliseconds);
                LastControllerTotalMilliseconds = controllerTotalAverage.Last;
                AverageControllerTotalMilliseconds = controllerTotalAverage.Average;
            }

            if(outcome.Status == VmRunStatus.Completed)
                CompletedFrames++;
            else
                FailedFrames++;
        }
    }

    #endregion

    #region Private helpers

    void UpdateInstructionAveragesLocked(VmRunTrace? trace)
    {
        if(trace is null)
            return;

        foreach(OpTraceEntry entry in trace.Instructions)
        {
            string key = $"{entry.Index:00} {entry.Operation}";
            if(!instructionAverages.TryGetValue(key, out MovingAverage? average))
            {
                average = new MovingAverage(60);
                instructionAverages.Add(key, average);
            }

            average.Add(entry.ElapsedMilliseconds);
        }
    }

    #endregion
}
