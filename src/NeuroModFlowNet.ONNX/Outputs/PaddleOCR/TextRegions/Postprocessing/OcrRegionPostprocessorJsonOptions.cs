namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly domain and geometry options for OCR text-region postprocessing.
/// RU: JSON-совместимые доменные и геометрические настройки постобработки OCR text-region.
/// </summary>
/// <remarks>
/// EN: Runtime options use PositiveInfinity for disabled maximum limits, which is awkward in JSON. This DTO keeps
/// disabled maxima as null and materializes them only when creating the hot-path value object.
/// RU: Runtime-настройки используют PositiveInfinity для выключенных максимумов, что неудобно в JSON. Этот DTO хранит
/// выключенные максимумы как null и превращает их в hot-path value object только при инициализации.
/// </remarks>
public sealed class OcrRegionPostprocessorJsonOptions
{
    public float MinRegionHeight { get; set; }

    public float? MaxRegionHeight { get; set; }

    public float MinRegionWidth { get; set; }

    public float? MaxRegionWidth { get; set; }

    public float MinRegionAspectRatio { get; set; }

    public float? MaxRegionAspectRatio { get; set; }

    public int? MaxRegions { get; set; }

    public OcrOverlapSuppressionOptions OverlapSuppression { get; set; } = new();

    public OcrLineMergeOptions LineMerge { get; set; } = new();

    public OcrRegionPostprocessorOptions ToRuntimeOptions(float? maxMergedRegionAspectRatioFallback = null) =>
        new()
        {
            MinRegionHeight = Math.Max(0, MinRegionHeight),
            MaxRegionHeight = ToOptionalMaximum(MaxRegionHeight),
            MinRegionWidth = Math.Max(0, MinRegionWidth),
            MaxRegionWidth = ToOptionalMaximum(MaxRegionWidth),
            MinRegionAspectRatio = Math.Max(0, MinRegionAspectRatio),
            MaxRegionAspectRatio = ToOptionalMaximum(MaxRegionAspectRatio),
            MaxRegions = MaxRegions is > 0 ? MaxRegions.Value : int.MaxValue,
            EnableOverlapSuppression = OverlapSuppression.Enabled,
            OverlapSuppressionRatio = Math.Clamp(OverlapSuppression.Ratio, 0f, 1f),
            EnableLineMerge = LineMerge.Enabled,
            MergeAngleDeltaDegrees = Math.Clamp(LineMerge.AngleDeltaDegrees, 0f, 90f),
            MergeNormalOffsetInHeights = Math.Max(0, LineMerge.NormalOffsetInHeights),
            MergeHeightRatio = Math.Max(1f, LineMerge.HeightRatio),
            MergeGapInHeights = Math.Max(0, LineMerge.GapInHeights),
            MinimumMergedCoverageRatio = Math.Clamp(LineMerge.MinimumCoverageRatio, 0f, 1f),
            MaxMergedRegionWidth = ToOptionalMaximum(LineMerge.MaxMergedRegionWidth),
            MaxMergedRegionAspectRatio = ToOptionalMaximum(LineMerge.MaxMergedRegionAspectRatio, maxMergedRegionAspectRatioFallback),
        };

    private static float ToOptionalMaximum(float? value, float? fallback = null)
    {
        float? resolvedValue = value ?? fallback;
        return resolvedValue is > 0 ? resolvedValue.Value : float.PositiveInfinity;
    }
}
