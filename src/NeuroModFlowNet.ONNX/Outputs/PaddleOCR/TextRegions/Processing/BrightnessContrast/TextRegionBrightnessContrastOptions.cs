namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly parameters for brightness/contrast ROI normalization.
/// RU: JSON-совместимые параметры нормализации яркости и контраста ROI.
/// </summary>
public sealed class TextRegionBrightnessContrastOptions : ITextRegionBrightnessContrastSettings
{
    public double Brightness { get; set; }

    public double ContrastPercent { get; set; } = 100;
}
