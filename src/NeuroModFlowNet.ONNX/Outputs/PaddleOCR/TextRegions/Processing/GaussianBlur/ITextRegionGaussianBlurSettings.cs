namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable Gaussian blur settings exposed separately from the processing implementation.
/// RU: Изменяемые настройки Gaussian blur, отделенные от реализации обработки.
/// </summary>
public interface ITextRegionGaussianBlurSettings
{
    int KernelSize { get; set; }

    double Sigma { get; set; }
}
