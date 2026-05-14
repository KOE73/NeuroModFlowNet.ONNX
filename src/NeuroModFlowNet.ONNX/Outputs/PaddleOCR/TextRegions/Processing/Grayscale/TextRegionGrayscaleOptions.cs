namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly channel selection for grayscale ROI conversion.
/// RU: JSON-совместимый выбор каналов для grayscale-преобразования ROI.
/// </summary>
public sealed class TextRegionGrayscaleOptions : ITextRegionGrayscaleSettings
{
    public bool UseRed { get; set; } = true;

    public bool UseGreen { get; set; } = true;

    public bool UseBlue { get; set; } = true;
}
