using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class OnnxExecutionBackendMatrix
{
    public static IEnumerable<InferenceBackend> EnabledBackends =>
        PipelineOnnxTestEnvironment.Current.EnabledExecutionBackends;
}
