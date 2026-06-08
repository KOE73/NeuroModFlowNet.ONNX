namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Immutable statistics snapshot for the OBB batch service.
/// </summary>
internal sealed record ObbBatchServiceStats(
    int BatchesProcessed,
    int TotalDetections,
    double LastBatchTimeMilliseconds,
    double AverageBatchTimeMilliseconds,
    double LastDetectionCount,
    double AverageDetectionCount,
    int CurrentQueueDepth)
{
    public static ObbBatchServiceStats Empty { get; } = new(
        BatchesProcessed: 0,
        TotalDetections: 0,
        LastBatchTimeMilliseconds: 0,
        AverageBatchTimeMilliseconds: 0,
        LastDetectionCount: 0,
        AverageDetectionCount: 0,
        CurrentQueueDepth: 0);
}
