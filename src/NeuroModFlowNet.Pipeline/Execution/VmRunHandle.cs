namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Execution handle representing a single started pipeline run.
/// RU: Дескриптор выполнения, представляющий один запущенный проход пайплайна.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Combines the identity of a run with a task tracking its completion and outcome.
/// Reasons: Allows the calling host code to await pipeline completion, read outputs, and inspect diagnostics for specific tasks in an asynchronous, non-blocking manner.
/// 
/// RU:
/// Суть: Объединяет идентификатор запуска с задачей отслеживания его завершения и результатов.
/// Причины: Позволяет вызывающему хост-коду асинхронно и без блокировок ожидать завершения пайплайна, 
///          получать выходные данные и анализировать диагностику конкретных задач.
/// </remarks>
public sealed class VmRunHandle
{
    internal VmRunHandle(VmRunIdentity identity, Task<VmRunOutcome> completion)
    {
        Identity = identity;
        Completion = completion;
    }

    public VmRunIdentity Identity { get; }

    public Task<VmRunOutcome> Completion { get; }
}

