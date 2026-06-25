namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Общая база для всех настроек провайдеров
/// TODO потом проанализировать и вытащить общие свойства
/// </summary>
public abstract class ExecutionProviderConfig
{
    public abstract InferenceBackend InferenceBackend { get; }

    public int DeviceId { get; set; } = 0;

    /// <summary>
    /// EN: Prevents ONNX Runtime from silently assigning unsupported nodes to the CPU Execution Provider.
    ///
    /// RU: Запрещает ONNX Runtime молча назначать неподдержанные nodes на CPU Execution Provider.
    /// </summary>
    /// <remarks>
    /// EN: Enabled by default because an explicitly selected GPU/backend mode is a placement contract. Set this to
    /// false only when partial provider coverage is intentional and the performance impact is understood.
    ///
    /// RU: Включено по умолчанию, потому что явно выбранный GPU/backend mode является контрактом размещения. Выключать
    /// следует только когда частичное покрытие provider-ом намеренно и влияние на производительность понятно.
    /// </remarks>
    public bool DisableCpuEpFallback { get; set; } = true;

    //public string? CachePath { get; set; }

    // Метод, который каждый EP реализует по-своему
    public abstract Dictionary<string, string> ToDictionary();
}
