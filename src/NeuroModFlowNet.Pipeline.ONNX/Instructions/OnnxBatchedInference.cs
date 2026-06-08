using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Executes one typed ONNX batching resource and resumes the current VM transaction with its own result.
/// </summary>
/// <remarks>
/// Domain commands such as <c>Model_YoloObb</c> and <c>Model_YoloBox</c> wrap this class so VM programs expose
/// meaningful CV operations, while the batching/waiting mechanics stay shared and invisible to the transaction code.
/// </remarks>
public class OnnxBatchedInference<TInput, TOutput> : Model_Inference<TInput, TOutput>
{
    public OnnxBatchedInference(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        OnnxBatchedResourceBase<TInput, TOutput> resource)
        : base(descriptor, inputKey, outputKey, resource)
    {
    }
}
