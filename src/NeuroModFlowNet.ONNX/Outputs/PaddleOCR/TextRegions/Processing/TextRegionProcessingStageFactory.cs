namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Builds runtime OCR ROI processing stages from configuration DTOs.
/// RU: Создает runtime-шаги обработки OCR ROI из конфигурационных DTO.
/// </summary>
/// <remarks>
/// EN: Keep config materialization outside the hot path. JSON or another configuration system should call this once
/// during module initialization, then reuse the returned stage/pipeline for frames.
/// RU: Материализацию config держим вне hot path. JSON или другая система конфигурации должна вызвать это один раз
/// при инициализации модуля, а затем переиспользовать возвращенный stage/pipeline на кадрах.
/// </remarks>
public static class TextRegionProcessingStageFactory
{
    public static ITextRegionProcessingStage CreateStage(TextRegionProcessingStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        return options.Kind switch
        {
            TextRegionProcessingStageKind.BrightnessContrast => new TextRegionBrightnessContrastStage(options.BrightnessContrast ?? new TextRegionBrightnessContrastOptions()),
            TextRegionProcessingStageKind.Gamma => new TextRegionGammaCorrectionStage(options.Gamma ?? new TextRegionGammaCorrectionOptions()),
            TextRegionProcessingStageKind.Grayscale => new TextRegionGrayscaleStage(options.Grayscale ?? new TextRegionGrayscaleOptions()),
            TextRegionProcessingStageKind.Threshold => new TextRegionThresholdStage(options.Threshold ?? new TextRegionThresholdOptions()),
            TextRegionProcessingStageKind.GaussianBlur => new TextRegionGaussianBlurStage(options.GaussianBlur ?? new TextRegionGaussianBlurOptions()),
            TextRegionProcessingStageKind.Sharpen => new TextRegionSharpenStage(options.Sharpen ?? new TextRegionSharpenOptions()),
            TextRegionProcessingStageKind.AutoContrast => new TextRegionAutoContrastStage(),
            _ => throw new ArgumentOutOfRangeException(nameof(options), options.Kind, "Unknown OCR ROI processing stage kind."),
        };
    }

    public static TextRegionProcessingPipeline CreatePipeline(IEnumerable<TextRegionProcessingStageOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);

        return new TextRegionProcessingPipeline([.. options.Select(CreateStage)]);
    }

    public static ITextRegionProcessingStage? CreateProcessingStage(TextRegionProcessingPipelineOptions? options)
    {
        if(options is null || !options.Enabled || options.Stages.Count == 0)
            return null;

        return CreatePipeline(options.Stages);
    }
}
