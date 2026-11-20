namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Factories for the built-in FFmpeg destinations. Additional sinks implement IVideoPacketSink.
///
/// RU: Фабрики встроенных выходов FFmpeg. Дополнительные выходы реализуют IVideoPacketSink.
/// </summary>
public static class FfmpegVideoOutputs
{
    public static VideoPacketOutput Mp4(string id, string filePath, int ioTimeoutMilliseconds = 5000)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string absolutePath = Path.GetFullPath(filePath);
        if(!string.Equals(Path.GetExtension(absolutePath), ".mp4", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("MP4 output requires an .mp4 file.", nameof(filePath));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ioTimeoutMilliseconds);
        return new(id, () => new FfmpegVideoPacketSink("mp4", absolutePath, ioTimeoutMilliseconds, network: false));
    }

    public static VideoPacketOutput Rtsp(string id, string destination, int ioTimeoutMilliseconds = 5000)
    {
        if(!Uri.TryCreate(destination, UriKind.Absolute, out Uri? uri) || uri.Scheme != "rtsp" || string.IsNullOrEmpty(uri.Host))
            throw new ArgumentException("An absolute RTSP publish URL is required.", nameof(destination));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ioTimeoutMilliseconds);
        return new(id, () => new FfmpegVideoPacketSink("rtsp", destination, ioTimeoutMilliseconds, network: true), Reconnect: true);
    }
}
