using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public sealed class OcrRoiBrightnessContrastStage : IOcrRoiProcessingStage, IOcrRoiBrightnessContrastSettings
{
    public string Name { get; } = "Brightness/Contrast";

    public double Brightness { get; set; }

    public double ContrastPercent { get; set; }

    public OcrRoiBrightnessContrastStage(double brightness, double contrastPercent)
    {
        Brightness = brightness;
        ContrastPercent = contrastPercent;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new Mat();
        source.ConvertTo(result, source.Type(), ContrastPercent / 100.0, Brightness);
        return result;
    }
}
