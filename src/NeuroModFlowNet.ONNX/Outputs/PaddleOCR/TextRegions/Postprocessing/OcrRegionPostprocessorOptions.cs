namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Domain and geometry options for OCR text-region postprocessing.
/// RU: Доменные и геометрические настройки постобработки OCR text-region.
/// </summary>
/// <remarks>
/// EN: The defaults keep the processor conservative: only invalid geometry is rejected, while overlap cleanup
/// and line merge stay opt-in. Realtime applications should tighten min/max size and max-count values from
/// their camera scale and recognition model width instead of relying on detector output alone.
/// RU: Значения по умолчанию оставляют процессор осторожным: отбрасывается только некорректная геометрия,
/// а удаление пересечений и склейка строк включаются явно. Realtime-приложениям стоит задавать min/max размеры
/// и max-count из масштаба камеры и ширины recognition-модели, а не полагаться только на выход детектора.
/// </remarks>
public readonly record struct OcrRegionPostprocessorOptions
{
    public float MinRegionHeight { get; init; } = 0f;
    public float MaxRegionHeight { get; init; } = float.PositiveInfinity;
    public float MinRegionWidth { get; init; } = 0f;
    public float MaxRegionWidth { get; init; } = float.PositiveInfinity;
    public float MinRegionAspectRatio { get; init; } = 0f;
    public float MaxRegionAspectRatio { get; init; } = float.PositiveInfinity;
    public int MaxRegions { get; init; } = int.MaxValue;
    public bool EnableOverlapSuppression { get; init; }
    public float OverlapSuppressionRatio { get; init; } = 0.65f;
    public bool EnableLineMerge { get; init; }
    public float MergeAngleDeltaDegrees { get; init; } = 12f;
    public float MergeNormalOffsetInHeights { get; init; } = 0.75f;
    public float MergeHeightRatio { get; init; } = 1.8f;
    public float MergeGapInHeights { get; init; } = 2.5f;
    public float MinimumMergedCoverageRatio { get; init; } = 0.45f;
    public float MaxMergedRegionWidth { get; init; } = float.PositiveInfinity;
    public float MaxMergedRegionAspectRatio { get; init; } = float.PositiveInfinity;

    public OcrRegionPostprocessorOptions()
    {
    }
}
