namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Общая база для всех настроек провайдеров
/// TODO потом проанализировать и вытащить общие свойства
/// </summary>
public abstract class ExecutionProviderConfig
{
    public abstract InferenceBackend InferenceBackend { get; }

    public int DeviceId { get; set; } = 0;
    //public string? CachePath { get; set; }

    // Метод, который каждый EP реализует по-своему
    public abstract Dictionary<string, string> ToDictionary();
}
