namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Per-connection adapter for the compressed stream. Open, Write and Dispose execute on the output worker.
/// A sink may retain a separate reference to a packet if it owns an additional asynchronous stage.
///
/// RU: Адаптер сжатого потока на одно подключение. Open, Write и Dispose выполняет воркер выхода.
/// Для асинхронной обработки адаптер может удержать отдельную ссылку на пакет.
/// </summary>
public interface IVideoPacketSink : IDisposable
{
    void Open(EncodedVideoStream stream, CancellationToken cancellationToken);
    void Write(EncodedVideoPacket packet, CancellationToken cancellationToken);
}
