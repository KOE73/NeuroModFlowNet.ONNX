namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Configuration options defining runtime policies for a VM controller instance.
/// RU: Параметры конфигурации, определяющие правила выполнения для экземпляра контроллера VM.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Groups VM settings, concurrency limits, initial ID offsets, output filter keys, and optional debugger hooks.
/// Reasons: Placing execution constraints (like MaxInFlight) at the controller level allows limiting the number of parallel VM contexts in memory, preventing system resource exhaustion without tangling individual domain instructions with queue/threading management.
/// 
/// RU:
/// Суть: Группирует настройки VM, лимиты параллелизма, начальное смещение RunId, ключи фильтрации выходов и хуки отладчика.
/// Причины: Вынесение ограничений выполнения (таких как MaxInFlight) на уровень контроллера позволяет лимитировать количество одновременно находящихся в памяти контекстов VM, предотвращая исчерпание системных ресурсов без усложнения отдельных бизнес-инструкций логикой управления очередями или потоками.
/// </remarks>
public sealed record VmControllerOptions(
    string SourceId,
    int MaxInFlight = 1,
    long InitialRunId = 0,
    IReadOnlyCollection<string>? OutputKeys = null,
    IOpDebugGate? DebugGate = null,
    bool RequireWarmup = false,
    bool CaptureVariablesInTrace = true)
{
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(SourceId);

        if(MaxInFlight <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxInFlight), "MaxInFlight must be positive.");

        if(InitialRunId < 0)
            throw new ArgumentOutOfRangeException(nameof(InitialRunId), "InitialRunId must be non-negative.");

        if(OutputKeys is null)
            return;

        foreach(string outputKey in OutputKeys)
            ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
    }
}
