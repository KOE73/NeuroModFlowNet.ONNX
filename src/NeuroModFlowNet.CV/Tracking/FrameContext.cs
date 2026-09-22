namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Neutral frame identity passed from the VM into tracking algorithms.
/// </summary>
public readonly record struct FrameContext(
    long RunId,
    DateTimeOffset Timestamp,
    long? SourceFrameId);
