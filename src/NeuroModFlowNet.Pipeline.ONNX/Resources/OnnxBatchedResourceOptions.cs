namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Batching policy for one ONNX-backed compute resource.
/// </summary>
/// <remarks>
/// Batch size, wait time, and pending capacity expose the throughput/latency/memory tradeoff explicitly instead of
/// hiding it inside a model command.
/// </remarks>
public sealed record OnnxBatchedResourceOptions(
    int MaxBatchSize = 1,
    TimeSpan? MaxWaitTime = null,
    int MaxPendingRequests = 32)
{
    public TimeSpan EffectiveMaxWaitTime => MaxWaitTime ?? TimeSpan.Zero;

    public void Validate()
    {
        if(MaxBatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxBatchSize), "MaxBatchSize must be positive.");

        if(EffectiveMaxWaitTime < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(MaxWaitTime), "MaxWaitTime must be non-negative.");

        if(MaxPendingRequests <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxPendingRequests), "MaxPendingRequests must be positive.");
    }
}

