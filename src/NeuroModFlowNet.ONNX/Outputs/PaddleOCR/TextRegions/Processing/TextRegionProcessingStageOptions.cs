namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: One JSON-friendly OCR ROI processing stage definition.
/// RU: Одно JSON-совместимое описание шага обработки OCR ROI.
/// </summary>
/// <remarks>
/// EN: The wrapper keeps the JSON contract simple for System.Text.Json: an array can deserialize without a custom
/// polymorphic converter, while each concrete processing type still has its own strongly typed parameter object.
/// RU: Обертка сохраняет простой JSON-контракт для System.Text.Json: массив десериализуется без кастомного
/// polymorphic converter, а у каждого типа обработки остаются свои строго типизированные параметры.
/// </remarks>
public sealed class TextRegionProcessingStageOptions
{
    public TextRegionProcessingStageKind Kind { get; set; }

    public TextRegionBrightnessContrastOptions? BrightnessContrast { get; set; }

    public TextRegionGammaCorrectionOptions? Gamma { get; set; }

    public TextRegionGrayscaleOptions? Grayscale { get; set; }

    public TextRegionThresholdOptions? Threshold { get; set; }

    public TextRegionGaussianBlurOptions? GaussianBlur { get; set; }

    public TextRegionSharpenOptions? Sharpen { get; set; }
}
