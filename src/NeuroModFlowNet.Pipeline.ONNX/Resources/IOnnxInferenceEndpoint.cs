using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Common pipeline-facing endpoint for one ONNX inference step.
/// </summary>
/// <remarks>
/// A VM instruction should not care whether the model is executed inline through a private runner, through a reloadable
/// runner, or through a batching service shared by many VM runs. The endpoint owns that execution policy and returns the
/// value that belongs to the current transaction.
/// </remarks>
public interface IOnnxInferenceEndpoint<in TInput, TOutput>
{
    string Name { get; }

    ValueTask<TOutput> InferAsync(
        VmRunIdentity identity,
        TInput input,
        CancellationToken cancellationToken);
}
