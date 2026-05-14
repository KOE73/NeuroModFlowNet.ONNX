namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly options for merging nearby OCR components into line-like regions.
/// RU: JSON-совместимые настройки склейки близких OCR-компонентов в line-like регионы.
/// </summary>
public sealed class OcrLineMergeOptions
{
    public bool Enabled { get; set; }

    public float AngleDeltaDegrees { get; set; } = 12f;

    public float NormalOffsetInHeights { get; set; } = 0.75f;

    public float HeightRatio { get; set; } = 1.8f;

    public float GapInHeights { get; set; } = 2.5f;

    public float MinimumCoverageRatio { get; set; } = 0.45f;

    public float? MaxMergedRegionWidth { get; set; }

    public float? MaxMergedRegionAspectRatio { get; set; }
}
