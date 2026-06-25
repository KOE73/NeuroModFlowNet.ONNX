namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal sealed class PipelineOnnxExecutionBackendSettings
{
    public bool Cpu { get; init; } = true;

    public bool Cuda { get; init; }

    public bool TensorRt { get; init; }
}
