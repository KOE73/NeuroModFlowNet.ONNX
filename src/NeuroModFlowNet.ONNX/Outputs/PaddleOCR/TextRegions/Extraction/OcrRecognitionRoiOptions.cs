namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly OCR recognition ROI size, height expansion, and image-processing options.
/// RU: JSON-совместимые настройки размера OCR ROI, расширения высоты и обработки изображения.
/// </summary>
/// <remarks>
/// EN: This is a configuration DTO, not the per-call extraction value. Materialize the processing pipeline once during
/// module initialization, then pass the returned stage into <see cref="CreateExtractionOptions"/>.
/// RU: Это конфигурационный DTO, а не value для каждого вызова вырезалки. Pipeline обработки надо материализовать один
/// раз при инициализации модуля, затем передать полученный stage в <see cref="CreateExtractionOptions"/>.
/// </remarks>
public sealed class OcrRecognitionRoiOptions
{
    public int TargetWidth { get; set; } = 320;

    public int TargetHeight { get; set; } = 48;

    public float HeightScale { get; set; } = 2f;

    public bool AdaptiveHeightEnabled { get; set; } = true;

    public float AdaptiveBasePad { get; set; } = 1f;

    public float AdaptivePadRatio { get; set; } = 0.25f;

    public float AdaptiveMaxPad { get; set; } = 8f;

    public TextRegionProcessingPipelineOptions? Processing { get; set; }

    public int NormalizedTargetWidth => Math.Max(1, TargetWidth);

    public int NormalizedTargetHeight => Math.Max(1, TargetHeight);

    public TextRegionExtractionOptions CreateExtractionOptions(ITextRegionProcessingStage? processingStage = null) =>
        new(NormalizedTargetWidth, NormalizedTargetHeight, processingStage);

    public ITextRegionProcessingStage? CreateProcessingStage() =>
        TextRegionProcessingStageFactory.CreateProcessingStage(Processing);

    public float CalculateHeightScale(float sourceRegionHeight)
    {
        if(!AdaptiveHeightEnabled)
            return Math.Max(0.1f, HeightScale);

        if(sourceRegionHeight <= 0)
            return Math.Max(0.1f, HeightScale);

        float pad = Math.Min(Math.Max(0, AdaptiveMaxPad), Math.Max(0, AdaptiveBasePad) + sourceRegionHeight * Math.Max(0, AdaptivePadRatio));
        float scaledHeight = sourceRegionHeight + pad * 2f;
        return Math.Max(0.1f, scaledHeight / sourceRegionHeight);
    }
}
