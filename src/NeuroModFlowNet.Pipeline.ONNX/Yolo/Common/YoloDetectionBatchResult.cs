namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Compact CPU-side detection result written back to a VM transaction.
/// </summary>
/// <remarks>
/// Large ONNX tensors and intermediate GPU payloads stay inside ONNX resources. The VM receives only the domain result
/// that later CPU logic, tracker, or branch instructions actually need.
/// </remarks>
public sealed record YoloDetectionBatchResult<TDetection>(
    IReadOnlyList<TDetection> Detections);

