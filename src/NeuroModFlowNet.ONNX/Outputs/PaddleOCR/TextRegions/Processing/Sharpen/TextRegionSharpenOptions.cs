namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly parameters for unsharp-mask ROI sharpening.
/// RU: JSON-совместимые параметры повышения резкости ROI через unsharp-mask.
/// </summary>
public sealed class TextRegionSharpenOptions : ITextRegionSharpenSettings
{
    public int KernelSize { get; set; } = 3;

    public double Sigma { get; set; }

    public double Amount { get; set; } = 1.0;
}
