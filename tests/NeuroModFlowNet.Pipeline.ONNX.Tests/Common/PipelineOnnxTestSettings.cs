namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal sealed class PipelineOnnxTestSettings
{
    public List<string> NativeLibrarySearchPaths { get; init; } = [];

    public string? CudaBinPath { get; init; }

    public string? CudnnBinPath { get; init; }

    public string? TrtLibPath { get; init; }

    public string? VisualArtifactsRoot { get; init; }

    public bool SaveVisualArtifacts { get; init; }

    public bool InteractiveVisualArtifacts { get; init; }

    public PipelineOnnxExecutionBackendSettings ExecutionBackends { get; init; } = new();
}
