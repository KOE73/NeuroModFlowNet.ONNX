namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Reads named registers from the active render run.
///
/// RU: Читает именованные регистры текущего запуска программы рисования.
/// </summary>
public interface IVmRegisterReader
{
    T Get<T>(string key);
    bool TryGet<T>(string key, out T value);
}
