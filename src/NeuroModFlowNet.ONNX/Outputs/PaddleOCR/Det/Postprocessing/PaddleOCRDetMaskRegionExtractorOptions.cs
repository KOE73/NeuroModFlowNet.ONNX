namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: DB-style PaddleOCR detection mask postprocessing options.
/// RU: Настройки DB-style постобработки mask-выхода PaddleOCR detection.
/// </summary>
/// <remarks>
/// EN: This option set is intentionally separate from OCR region postprocessing. Mask extraction owns
/// threshold, contour scoring and unclip parameters, while the common OCR region postprocessor owns
/// domain filters, overlap cleanup and line merge after regions already exist.
/// RU: Этот набор настроек намеренно отделен от OCR region postprocessing. Извлечение из mask владеет
/// threshold, contour scoring и unclip-параметрами, а общий OCR region postprocessor владеет доменными
/// фильтрами, удалением пересечений и склейкой строк уже после появления регионов.
/// </remarks>
public readonly record struct PaddleOCRDetMaskRegionExtractorOptions
{
    public float BitmapThreshold { get; init; } = 0.3f;
    public float BoxScoreThreshold { get; init; } = 0.7f;
    public float MinimumBoxSide { get; init; } = 3f;
    public int MaxCandidateCount { get; init; } = 1000;
    public bool EnableUnclip { get; init; } = true;
    public float UnclipRatio { get; init; } = 2f;

    public PaddleOCRDetMaskRegionExtractorOptions()
    {
    }
}
