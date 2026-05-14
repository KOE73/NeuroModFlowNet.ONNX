using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Converts OCR text-region crops to black/white while keeping BGR output for the recognition converter.
/// RU: Переводит OCR-кропы в черно-белый вид, но возвращает BGR-изображение для recognition converter.
/// </summary>
public sealed class TextRegionThresholdStage : ITextRegionProcessingStage, ITextRegionThresholdSettings
{
    public string Name { get; } = "Threshold";

    public double Threshold { get; set; } = 128;

    public bool UseOtsu { get; set; }

    public TextRegionThresholdStage()
    {
    }

    public TextRegionThresholdStage(TextRegionThresholdOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        Threshold = options.Threshold;
        UseOtsu = options.UseOtsu;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using Mat gray = ToGray(source);
        using var binary = new Mat();

        ThresholdTypes thresholdType = ThresholdTypes.Binary;
        double threshold = Math.Clamp(Threshold, 0, 255);
        if(UseOtsu)
        {
            threshold = 0;
            thresholdType |= ThresholdTypes.Otsu;
        }

        Cv2.Threshold(gray, binary, threshold, 255, thresholdType);

        var result = new Mat();
        Cv2.CvtColor(binary, result, ColorConversionCodes.GRAY2BGR);
        return result;
    }

    private static Mat ToGray(Mat source)
    {
        if(source.Channels() == 1)
            return source.Clone();

        var gray = new Mat();
        Cv2.CvtColor(source, gray, ColorConversionCodes.BGR2GRAY);
        return gray;
    }
}
