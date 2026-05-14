using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Fast automatic contrast stretch for OCR text-region crops.
/// RU: Быстрое автоматическое растяжение контраста для OCR-кропов текстовых областей.
/// </summary>
public sealed class TextRegionAutoContrastStage : ITextRegionProcessingStage
{
    public string Name { get; } = "AutoContrast";

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new Mat();
        Cv2.Normalize(source, result, 0, 255, NormTypes.MinMax, source.Type().Value);
        return result;
    }
}
