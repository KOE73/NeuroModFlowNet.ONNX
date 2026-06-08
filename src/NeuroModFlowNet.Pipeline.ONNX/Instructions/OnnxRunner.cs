using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Executes one typed ONNX runner resource with serialized access to the underlying runner instance.
/// </summary>
/// <remarks>
/// This is a technical adapter for cases where a model-specific command does not need custom result shaping. Public VM
/// programs should normally expose domain names such as <c>Model_PaddleDet</c>, while this class keeps the repeated
/// "read input, call resource, write output" mechanics in one place.
/// </remarks>
public class OnnxRunner<TInput, TOutput> : Model_Inference<TInput, TOutput>
{
    public OnnxRunner(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        OnnxRunnerResource<TInput, TOutput> resource,
        bool disposeOutputWithContext = false)
        : base(descriptor, inputKey, outputKey, resource, disposeOutputWithContext)
    {
    }
}
