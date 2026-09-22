using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a per-source tracker over neutral detections.
/// </summary>
public sealed class Op_Track : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly string trackerStateKey;
    readonly IouTrackerOptions trackerOptions;
    readonly string? startZoneKey;
    readonly string? endZoneKey;

    public Op_Track(
        string inputKey,
        string outputKey,
        string trackerStateKey,
        string syncGate,
        IouTrackerOptions trackerOptions,
        string? startZoneKey = null,
        string? endZoneKey = null)
        : base(OpDescriptor.Create(
            "Op_Track",
            "op.track",
            [VarRequirement.Read<TrackDetection[]>(inputKey)],
            [VarRequirement.Write<TrackedObject[]>(outputKey)],
            hasSideEffects: true,
            syncGate: syncGate))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(trackerStateKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(syncGate);
        trackerOptions.Validate();

        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.trackerStateKey = trackerStateKey;
        this.trackerOptions = trackerOptions;
        this.startZoneKey = startZoneKey;
        this.endZoneKey = endZoneKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        TrackDetection[] detections = context.Get<TrackDetection[]>(inputKey);
        ITracker tracker = context.GlobalMemory.GetOrAdd<ITracker>(
            trackerStateKey,
            _ => new IouTracker(trackerOptions));

        var frame = new FrameContext(
            context.Identity.RunId,
            context.Identity.Timestamp,
            context.Identity.SourceFrameId);
        var frameOptions = new TrackerFrameOptions(
            ReadOptionalGlobal<TrackRect>(context, startZoneKey),
            ReadOptionalGlobal<TrackRect>(context, endZoneKey));

        IReadOnlyList<TrackedObject> trackedObjects = tracker.Process(detections, frame, frameOptions);
        context.Set(outputKey, trackedObjects.ToArray());
        return ValueTask.FromResult(OpResult.Continue);
    }

    static T? ReadOptionalGlobal<T>(VmRunContext context, string? key)
        where T : struct
    {
        if(string.IsNullOrWhiteSpace(key))
            return null;

        if(!context.GlobalMemory.TryGet(key, out T value))
            return null;

        return value;
    }
}
