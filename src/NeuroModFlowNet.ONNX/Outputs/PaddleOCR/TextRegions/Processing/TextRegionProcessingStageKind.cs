using System.Text.Json.Serialization;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Stable configuration discriminator for OCR ROI image-processing stages.
/// RU: Стабильный конфигурационный тип шага обработки OCR ROI.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum TextRegionProcessingStageKind
{
    BrightnessContrast,
    Gamma,
    Grayscale,
    Threshold,
    GaussianBlur,
    Sharpen,
    AutoContrast
}
