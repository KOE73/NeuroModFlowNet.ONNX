namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Terminal result and metadata captured after a pipeline execution finishes.
/// RU: Конечный результат и метаданные, полученные после завершения выполнения пайплайна.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Groups identity, final status, outputs, diagnostics trace, and any failure exceptions.
/// Reasons: Returning a structured outcome instead of throwing unhandled background task exceptions prevents application crashes. It ensures that the host can gracefully process successes/failures and examine step-by-step diagnostics for every run.
/// 
/// RU:
/// Суть: Группирует идентификатор, финальный статус, выходные данные, диагностический лог и возникшие исключения.
/// Причины: Возврат структурированного результата вместо выброса необработанных исключений в фоновых задачах 
///          предотвращает падение хост-приложения. Это гарантирует, что вызывающая сторона сможет корректно 
///          обработать любой исход (успех или сбой) и изучить пошаговую диагностику по каждому запуску.
/// </remarks>
public sealed record VmRunOutcome(
    VmRunIdentity Identity,
    VmRunStatus Status,
    Exception? Exception = null,
    VmRunOutput? Output = null,
    VmRunTrace? Trace = null,
    VmRunTiming? Timing = null);
