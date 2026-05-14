namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly parameters for binary ROI thresholding.
/// RU: JSON-совместимые параметры бинаризации ROI.
/// </summary>
public sealed class TextRegionThresholdOptions : ITextRegionThresholdSettings
{
    public double Threshold { get; set; } = 128;

    public bool UseOtsu { get; set; }
}
