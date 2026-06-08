namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Assigns dense sequential run IDs and records state changes of pipeline executions.
/// RU: Назначает плотные последовательные идентификаторы запусков (RunId) и регистрирует изменения состояния выполнения.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Acts as a thread-safe journal/ledger of all pipeline runs processed by the VM controller.
/// Reasons: Tracks the lifecycle and outcomes of runs. Allows ordered synchronization points to determine if a preceding run failed or succeeded, avoiding infinite waits while preserving execution order.
/// 
/// RU:
/// Суть: Служит потокобезопасным журналом учета всех запусков пайплайна, переданных контроллеру VM.
/// Причины: Отслеживает жизненный цикл и статус завершения запусков. 
///          Это позволяет точкам упорядоченной синхронизации определять, завершился ли предыдущий запуск (успешно или с ошибкой),
///          предотвращая бесконечные блокировки ожидания и сохраняя хронологический порядок.
/// </remarks>
public sealed class VmRunLedger
{
    readonly object syncRoot = new();
    readonly Dictionary<long, VmRunRecord> records = [];
    long nextRunId;

    public VmRunLedger(long initialRunId = 0)
    {
        if(initialRunId < 0)
            throw new ArgumentOutOfRangeException(nameof(initialRunId), "Initial run id must be non-negative.");

        nextRunId = initialRunId;
    }

    public VmRunIdentity Accept(string sourceId, DateTimeOffset timestamp, long? sourceFrameId = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceId);

        lock(syncRoot)
        {
            long runId = nextRunId++;
            var identity = new VmRunIdentity(sourceId, runId, timestamp, sourceFrameId);
            records.Add(runId, new VmRunRecord(identity, VmRunStatus.Accepted, null));
            return identity;
        }
    }

    public void MarkRunning(long runId) => Update(runId, VmRunStatus.Running, null);

    public void MarkCompleted(long runId) => Update(runId, VmRunStatus.Completed, null);

    public void MarkFailed(long runId, Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        Update(runId, VmRunStatus.Failed, exception);
    }

    public void MarkCanceled(long runId, Exception? exception = null) => Update(runId, VmRunStatus.Canceled, exception);

    public VmRunRecord? TryGet(long runId)
    {
        lock(syncRoot)
            return records.GetValueOrDefault(runId);
    }

    public IReadOnlyList<VmRunRecord> Snapshot()
    {
        lock(syncRoot)
            return records.Values.OrderBy(record => record.Identity.RunId).ToArray();
    }

    static bool IsTerminal(VmRunStatus status) =>
        status is VmRunStatus.Completed or VmRunStatus.Failed or VmRunStatus.Canceled;

    void Update(long runId, VmRunStatus status, Exception? exception)
    {
        lock(syncRoot)
        {
            if(!records.TryGetValue(runId, out VmRunRecord? current))
                throw new InvalidOperationException($"Run {runId} is not registered in this ledger.");

            if(IsTerminal(current.Status))
                return;

            records[runId] = current with
            {
                Status = status,
                Exception = exception
            };
        }
    }
}

