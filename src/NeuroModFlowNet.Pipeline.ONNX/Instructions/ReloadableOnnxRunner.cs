using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Executes an ONNX runner resource whose inner runner can be replaced without rebuilding the VM program.
/// </summary>
/// <remarks>
/// This adapter is intentionally separate from <see cref="OnnxRunner{TInput,TOutput}"/> because replacement has stronger
/// synchronization requirements: inference and runner swap must not overlap around the same native ONNX Runtime state.
/// </remarks>
public class ReloadableOnnxRunner<TInput, TOutput> : Model_Inference<TInput, TOutput>
{
    public ReloadableOnnxRunner(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        ReloadableOnnxRunnerResource<TInput, TOutput> resource,
        bool disposeOutputWithContext = false)
        : base(descriptor, inputKey, outputKey, resource, disposeOutputWithContext)
    {
    }
}
