namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Immutable snapshot of the state currently known for one accepted run.
/// RU: Неизменяемый снимок состояния, известного на данный момент для одного принятого к исполнению запуска.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Represents a ledger entry containing the identity, execution status, and optional exception details.
/// Reasons: Used to provide external callers or synchronization gates with a read-only snapshot of a run's state in a thread-safe manner, shielding internal mutable ledger structures.
/// 
/// RU:
/// Суть: Представляет собой запись в журнале учета, содержащую идентификатор, статус выполнения и сведения об исключении.
/// Причины: Используется для предоставления внешним вызывающим объектам или шлюзам синхронизации 
///          снимка состояния запуска только для чтения потокобезопасным способом, защищая внутренние изменяемые структуры журнала.
/// </remarks>
public sealed record VmRunRecord(
    VmRunIdentity Identity,
    VmRunStatus Status,
    Exception? Exception);

