namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Startup definition for a YOLO OBB resource.
/// </summary>
public sealed record YoloObbResourceDefinition(
    string Name,
    string ModelPath,
    YoloObbRunnerKind RunnerKind,
    OnnxBatchedResourceOptions Batching);

