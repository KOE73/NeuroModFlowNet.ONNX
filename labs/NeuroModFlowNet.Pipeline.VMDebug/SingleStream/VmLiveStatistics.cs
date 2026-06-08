namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Aggregates current and moving-average timing values for the single-stream Spectre dashboard.
/// </summary>
internal sealed class VmLiveStatistics
{
    readonly MovingAverage runTotalTime = new(60);
    readonly MovingAverage instructionTotalTime = new(60);
    readonly MovingAverage outsideTraceTime = new(60);
    readonly MovingAverage controllerTotalTime = new(60);
    readonly MovingAverage programTime = new(60);
    readonly MovingAverage outputCaptureTime = new(60);
    readonly MovingAverage contextDisposeTime = new(60);
    readonly MovingAverage taskScheduleGapTime = new(60);
    readonly Dictionary<string, MovingAverage> instructionTimes = new(StringComparer.Ordinal);
    readonly DateTimeOffset startedAt = DateTimeOffset.UtcNow;

    public int CameraIndex { get; private set; }

    public int CameraWidth { get; private set; }

    public int CameraHeight { get; private set; }

    public double CameraFps { get; private set; }

    public int CompletedFrames { get; private set; }

    public int FailedFrames { get; private set; }

    public TimeSpan Uptime => DateTimeOffset.UtcNow - startedAt;

    public double LastRunTotalMilliseconds => runTotalTime.Last;

    public double AverageRunTotalMilliseconds => runTotalTime.Average;

    public double LastInstructionTotalMilliseconds => instructionTotalTime.Last;

    public double AverageInstructionTotalMilliseconds => instructionTotalTime.Average;

    public double LastOutsideTraceMilliseconds => outsideTraceTime.Last;

    public double AverageOutsideTraceMilliseconds => outsideTraceTime.Average;

    public double ProcessingFps => runTotalTime.Average <= 0 ? 0 : 1000.0 / runTotalTime.Average;

    public double LastControllerTotalMilliseconds => controllerTotalTime.Last;

    public double AverageControllerTotalMilliseconds => controllerTotalTime.Average;

    public double LastProgramMilliseconds => programTime.Last;

    public double AverageProgramMilliseconds => programTime.Average;

    public double LastOutputCaptureMilliseconds => outputCaptureTime.Last;

    public double AverageOutputCaptureMilliseconds => outputCaptureTime.Average;

    public double LastContextDisposeMilliseconds => contextDisposeTime.Last;

    public double AverageContextDisposeMilliseconds => contextDisposeTime.Average;

    public double LastTaskScheduleGapMilliseconds => taskScheduleGapTime.Last;

    public double AverageTaskScheduleGapMilliseconds => taskScheduleGapTime.Average;

    public IReadOnlyList<InstructionTimingSnapshot> InstructionSnapshots =>
        instructionTimes
            .OrderBy(item => item.Key, StringComparer.Ordinal)
            .Select(item => new InstructionTimingSnapshot(item.Key, item.Value.Last, item.Value.Average))
            .ToArray();

    public void SetCamera(int cameraIndex, int width, int height, double fps)
    {
        CameraIndex = cameraIndex;
        CameraWidth = width;
        CameraHeight = height;
        CameraFps = fps;
    }

    public void AddWarmup(VmRunOutcome outcome)
    {
        AddInstructionSamples(outcome.Trace);
    }

    public void AddFrame(VmRunOutcome outcome, TimeSpan frameRunTotalTime)
    {
        double runTotalMilliseconds = frameRunTotalTime.TotalMilliseconds;
        double instructionTotalMilliseconds = outcome.Trace?.TotalInstructionMilliseconds ?? 0;

        runTotalTime.Add(runTotalMilliseconds);
        outsideTraceTime.Add(Math.Max(0, runTotalMilliseconds - instructionTotalMilliseconds));

        if(outcome.Timing is { } timing)
        {
            controllerTotalTime.Add(timing.TotalMilliseconds);
            programTime.Add(timing.ProgramMilliseconds);
            outputCaptureTime.Add(timing.OutputCaptureMilliseconds);
            contextDisposeTime.Add(timing.ContextDisposeMilliseconds);
            taskScheduleGapTime.Add(Math.Max(0, runTotalMilliseconds - timing.TotalMilliseconds));
        }

        if(outcome.Status == VmRunStatus.Completed)
            CompletedFrames++;
        else
            FailedFrames++;

        AddInstructionSamples(outcome.Trace);
    }

    void AddInstructionSamples(VmRunTrace? trace)
    {
        if(trace is null)
            return;

        instructionTotalTime.Add(trace.TotalInstructionMilliseconds);

        foreach(OpTraceEntry entry in trace.Instructions)
        {
            string key = $"{entry.Index:00} {entry.Operation}";
            if(!instructionTimes.TryGetValue(key, out MovingAverage? movingAverage))
            {
                movingAverage = new MovingAverage(60);
                instructionTimes.Add(key, movingAverage);
            }

            movingAverage.Add(entry.ElapsedMilliseconds);
        }
    }
}
