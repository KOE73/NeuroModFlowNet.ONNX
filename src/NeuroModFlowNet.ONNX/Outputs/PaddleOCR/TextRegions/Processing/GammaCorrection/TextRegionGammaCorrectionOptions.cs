namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly parameters for gamma correction.
/// RU: JSON-совместимые параметры gamma-коррекции.
/// </summary>
public sealed class TextRegionGammaCorrectionOptions : ITextRegionGammaCorrectionSettings
{
    public double Gamma { get; set; } = 1.0;
}
