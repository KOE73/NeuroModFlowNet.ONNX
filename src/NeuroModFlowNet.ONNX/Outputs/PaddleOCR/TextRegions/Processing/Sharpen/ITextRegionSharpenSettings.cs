namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable settings for OCR crop sharpening based on an unsharp-mask pass.
/// RU: Изменяемые настройки повышения резкости OCR-кропа через unsharp-mask проход.
/// </summary>
public interface ITextRegionSharpenSettings
{
    int KernelSize { get; set; }

    double Sigma { get; set; }

    double Amount { get; set; }
}
