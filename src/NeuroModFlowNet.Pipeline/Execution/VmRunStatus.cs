namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Describes the lifecycle states of one accepted VM execution.
/// RU: Описывает состояния жизненного цикла одного принятого к исполнению запуска в VM.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Enumerates the progression of execution from acceptance to terminal states (completed, failed, or canceled).
/// Reasons: Allows the controller, ledger, and synchronization gates to inspect and branch based on the exact phase of a run.
/// 
/// RU:
/// Суть: Перечисляет этапы выполнения — от принятия запуска до его завершения (успешно, со сбоем или отменено).
/// Причины: Позволяет контроллеру, журналу учета и шлюзам синхронизации отслеживать текущую фазу запуска и реагировать на неё.
/// </remarks>
public enum VmRunStatus
{
    Accepted,
    Running,
    Completed,
    Failed,
    Canceled
}

