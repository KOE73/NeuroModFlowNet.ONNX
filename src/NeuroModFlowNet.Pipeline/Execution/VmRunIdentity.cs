namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Identifies one accepted execution of a pipeline program for one source.
/// RU: Идентифицирует один принятый к исполнению запуск программы пайплайна для конкретного источника.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Groups routing metadata (SourceId, Timestamp) with a dense sequential execution index (RunId).
/// Reasons: RunId is intentionally decoupled from external frame IDs. Video capture loops may drop frames before processing, but the VM requires a continuous, dense sequence of run IDs to ensure that ordered synchronization gates do not stall waiting for skipped frames.
/// 
/// RU:
/// Суть: Группирует метаданные маршрутизации (SourceId, Timestamp) с плотным последовательным индексом исполнения (RunId).
/// Причины: RunId намеренно отделен от номеров кадров источника. 
///          Захват видео может пропускать кадры до начала обработки,
///          но для работы шлюзов упорядоченной синхронизации в VM требуется непрерывная плотная последовательность RunId, 
///          чтобы избежать зависаний при ожидании пропущенных кадров.
/// </remarks>
public readonly record struct VmRunIdentity(
    string SourceId,
    long RunId,
    DateTimeOffset Timestamp,
    long? SourceFrameId = null);

