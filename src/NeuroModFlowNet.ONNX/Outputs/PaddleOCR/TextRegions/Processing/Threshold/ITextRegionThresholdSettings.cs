namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable binary threshold settings exposed separately from the processing implementation.
/// RU: Изменяемые настройки бинаризации, отделенные от реализации обработки.
/// </summary>
public interface ITextRegionThresholdSettings
{
    double Threshold { get; set; }

    bool UseOtsu { get; set; }
}
