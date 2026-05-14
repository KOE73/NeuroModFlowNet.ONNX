namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly parameters for Gaussian blur ROI filtering.
/// RU: JSON-совместимые параметры Gaussian blur для ROI.
/// </summary>
public sealed class TextRegionGaussianBlurOptions : ITextRegionGaussianBlurSettings
{
    public int KernelSize { get; set; } = 3;

    public double Sigma { get; set; }
}
