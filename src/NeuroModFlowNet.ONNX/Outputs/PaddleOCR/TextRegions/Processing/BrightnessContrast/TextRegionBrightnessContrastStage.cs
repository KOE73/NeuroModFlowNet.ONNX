using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Simple brightness/contrast normalization stage for OCR text-region crops.
/// RU: Простой шаг нормализации яркости и контраста для OCR-кропов текстовых областей.
/// </summary>
/// <remarks>
/// EN: This remains a processing stage, not part of the extractor, because image normalization is data/model
/// dependent and should be easy to replace or disable in realtime experiments.
/// RU: Это остается processing stage, а не частью вырезалки, потому что нормализация изображения зависит от
/// данных и модели; в realtime-экспериментах ее должно быть легко заменить или отключить.
/// </remarks>
public sealed class TextRegionBrightnessContrastStage : ITextRegionProcessingStage, ITextRegionBrightnessContrastSettings
{
    public string Name { get; } = "Brightness/Contrast";

    public double Brightness { get; set; }

    public double ContrastPercent { get; set; } = 100;

    public TextRegionBrightnessContrastStage()
    {
    }

    public TextRegionBrightnessContrastStage(TextRegionBrightnessContrastOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        Brightness = options.Brightness;
        ContrastPercent = options.ContrastPercent;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new Mat();
        source.ConvertTo(result, source.Type(), ContrastPercent / 100.0, Brightness);
        return result;
    }
}
