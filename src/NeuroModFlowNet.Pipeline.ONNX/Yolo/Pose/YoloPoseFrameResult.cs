using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Single VM-request YOLO Pose result.
/// </summary>
public sealed class YoloPoseFrameResult
{
    public YoloPoseFrameResult(YoloPose[] detections)
    {
        Detections = detections ?? throw new ArgumentNullException(nameof(detections));
    }

    public YoloPose[] Detections { get; }
}
