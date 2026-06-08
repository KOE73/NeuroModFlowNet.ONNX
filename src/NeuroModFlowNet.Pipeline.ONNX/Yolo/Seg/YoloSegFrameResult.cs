using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Single VM-request YOLO Segmentation result.
/// </summary>
/// <remarks>
/// Existing YOLO Seg extractors expose single-frame results through types that still implement <see cref="IBatchedResult"/>
/// for historical reasons. The VM transaction should not depend on that batching marker, so the instruction wraps the
/// concrete payload here before publishing it to the program.
/// </remarks>
public sealed class YoloSegFrameResult
{
    public YoloSegFrameResult(YoloSegResult_FP32_Mask32 result)
    {
        FP32 = result ?? throw new ArgumentNullException(nameof(result));
    }

    public YoloSegFrameResult(YoloSegResult_FP16_Mask32 result)
    {
        FP16 = result ?? throw new ArgumentNullException(nameof(result));
    }

    public YoloSegResult_FP32_Mask32? FP32 { get; }

    public YoloSegResult_FP16_Mask32? FP16 { get; }
}
