namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable grayscale channel settings exposed separately from the processing implementation.
/// RU: Изменяемые настройки каналов grayscale, отделенные от реализации обработки.
/// </summary>
public interface ITextRegionGrayscaleSettings
{
    bool UseRed { get; set; }

    bool UseGreen { get; set; }

    bool UseBlue { get; set; }
}
