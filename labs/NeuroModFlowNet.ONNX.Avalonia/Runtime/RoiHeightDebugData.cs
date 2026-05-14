namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Compact diagnostics for the adaptive OCR ROI height formula shown next to each recognition preview.
/// RU: Компактная диагностика формулы adaptive OCR ROI height, которая показывается рядом с preview распознавания.
/// </summary>
public readonly record struct RoiHeightDebugData(
    float SourceHeight,
    float Pad,
    float Scale)
{
    public static RoiHeightDebugData Empty { get; } = new(0, 0, 0);
}
