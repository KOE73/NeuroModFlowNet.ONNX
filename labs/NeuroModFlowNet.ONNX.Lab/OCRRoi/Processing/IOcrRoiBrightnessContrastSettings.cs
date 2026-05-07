namespace NeuroModFlowNet.ONNX;

public interface IOcrRoiBrightnessContrastSettings
{
    double Brightness { get; set; }

    double ContrastPercent { get; set; }
}
