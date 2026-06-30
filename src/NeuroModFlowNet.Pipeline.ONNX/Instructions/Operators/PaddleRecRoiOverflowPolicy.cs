namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Policy applied when actual OCR ROI count exceeds the fixed output capacity.
///
/// RU: Политика для случая, когда фактическое число OCR ROI больше фиксированной емкости выхода.
/// </summary>
public enum PaddleRecRoiOverflowPolicy
{
    Fail,
    Truncate
}
