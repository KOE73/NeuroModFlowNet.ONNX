namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Startup definition for a YOLO Box resource.
/// </summary>
public sealed record YoloBoxResourceDefinition(
    string Name,
    string ModelPath,
    YoloBoxRunnerKind RunnerKind,
    OnnxBatchedResourceOptions Batching);

