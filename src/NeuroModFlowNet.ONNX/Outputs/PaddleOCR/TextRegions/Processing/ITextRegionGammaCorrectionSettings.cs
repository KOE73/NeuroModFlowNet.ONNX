namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable gamma setting exposed separately from the processing implementation.
/// RU: Изменяемая настройка gamma, вынесенная отдельно от реализации обработки.
/// </summary>
/// <remarks>
/// EN: The small contract keeps realtime controls independent from the current gamma implementation.
/// RU: Маленький контракт отделяет realtime-настройки от текущей реализации gamma-коррекции.
/// </remarks>
public interface ITextRegionGammaCorrectionSettings
{
    double Gamma { get; set; }
}
