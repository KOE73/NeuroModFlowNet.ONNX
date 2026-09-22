namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// CPU-only point used by tracking paths without binding the CV core to UI, OpenCV, or ONNX types.
/// </summary>
public readonly record struct TrackPoint(float X, float Y);
