namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Blocking frame producer owned by the host, never by the VM. Capture (files, RTSP, USB) stays outside
/// <c>NeuroModFlowNet.Pipeline</c> by design; the VM only sees the registers an <see cref="ISourceFrame"/> adds.
///
/// RU: Блокирующий источник кадров, принадлежащий хосту, а не VM. Захват (файлы, RTSP, USB) намеренно живёт вне
/// <c>NeuroModFlowNet.Pipeline</c>; VM видит только регистры, которые добавляет <see cref="ISourceFrame"/>.
/// </summary>
internal interface IFrameSource : IDisposable
{
    string Description { get; }

    /// <summary>Nominal source FPS used for playback pacing; 0 when unknown (live streams).</summary>
    double Fps { get; }

    /// <summary>
    /// Returns the next frame or <c>null</c> when the source is exhausted. The caller owns the returned frame.
    /// Implementations may block (network, reconnect backoff) and must honor <paramref name="cancellationToken"/>.
    /// </summary>
    ISourceFrame? ReadNext(CancellationToken cancellationToken);
}
