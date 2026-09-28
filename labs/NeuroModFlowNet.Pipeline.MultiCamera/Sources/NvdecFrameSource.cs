using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: NVDEC source: FFmpeg demux + hardware decode straight into CUDA memory (no host copy). Files play back to back
/// and loop; an RTSP stream is reopened with backoff after an error. Same playlist semantics as the CPU sources.
///
/// RU: NVDEC-источник: демультиплексор FFmpeg + аппаратный декод прямо в память CUDA (без копии в host). Файлы идут
/// подряд и зацикливаются; RTSP-поток после ошибки переоткрывается с паузой. Семантика плейлиста как у CPU-источников.
/// </summary>
internal sealed class NvdecFrameSource : IFrameSource
{
    static readonly TimeSpan MinBackoff = TimeSpan.FromSeconds(1);
    static readonly TimeSpan MaxBackoff = TimeSpan.FromSeconds(15);

    readonly IReadOnlyList<string> sources;
    readonly bool loop;
    readonly bool isLive;
    readonly int extraHardwareFrames;
    NvdecVideoDecoder? decoder;
    int sourceIndex = -1;
    TimeSpan backoff = MinBackoff;

    NvdecFrameSource(IReadOnlyList<string> sources, bool loop, bool isLive, int extraHardwareFrames, string description)
    {
        this.sources = sources;
        this.loop = loop;
        this.isLive = isLive;
        this.extraHardwareFrames = extraHardwareFrames;
        Description = description;

        // Open the first source now: FFmpeg must create its CUDA device on the primary context before ONNX Runtime
        // touches it, and a wrong path should fail at startup, not on the first frame.
        OpenNext();
    }

    public string Description { get; }

    public double Fps => isLive ? 0 : decoder?.Fps ?? 0;

    public static NvdecFrameSource FromFiles(IReadOnlyList<string> files, bool loop, int extraHardwareFrames)
    {
        foreach(string file in files)
        {
            if(!File.Exists(file))
                throw new FileNotFoundException($"Video file was not found: {file}", file);
        }

        return new NvdecFrameSource(files.ToArray(), loop, isLive: false, extraHardwareFrames,
            $"nvdec files [{string.Join(", ", files.Select(Path.GetFileName))}]{(loop ? " (loop)" : string.Empty)}");
    }

    public static NvdecFrameSource FromRtsp(string url, int extraHardwareFrames) =>
        new([url], loop: true, isLive: true, extraHardwareFrames, $"nvdec rtsp {MaskCredentials(url)}");

    public ISourceFrame? ReadNext(CancellationToken cancellationToken)
    {
        while(!cancellationToken.IsCancellationRequested)
        {
            try
            {
                decoder ??= OpenNext();
                if(decoder is null)
                    return null;

                NvdecFrame? frame = decoder.ReadNext();
                if(frame is not null)
                {
                    backoff = MinBackoff;
                    return new NvdecSourceFrame(frame);
                }

                // End of this file: continue with the next one of the playlist.
                CloseCurrent();
            }
            catch(InvalidOperationException) when(isLive)
            {
                // Live stream dropped: reopen after a growing pause. Files fail loudly instead.
                CloseCurrent();
                cancellationToken.WaitHandle.WaitOne(backoff);
                backoff = TimeSpan.FromTicks(Math.Min(backoff.Ticks * 2, MaxBackoff.Ticks));
            }
        }

        return null;
    }

    NvdecVideoDecoder? OpenNext()
    {
        sourceIndex++;
        if(sourceIndex >= sources.Count)
        {
            if(!loop)
                return null;

            sourceIndex = 0;
        }

        decoder = new NvdecVideoDecoder(sources[sourceIndex], loop: false, useFence: true, extraHardwareFrames: extraHardwareFrames);
        return decoder;
    }

    void CloseCurrent()
    {
        decoder?.Dispose();
        decoder = null;
    }

    static string MaskCredentials(string url)
    {
        int schemeEnd = url.IndexOf("://", StringComparison.Ordinal);
        int at = url.IndexOf('@');
        return schemeEnd >= 0 && at > schemeEnd
            ? string.Concat(url.AsSpan(0, schemeEnd + 3), "***", url.AsSpan(at))
            : url;
    }

    public void Dispose() => CloseCurrent();
}
