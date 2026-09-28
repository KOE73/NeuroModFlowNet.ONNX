using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Live source (RTSP URL or local camera index) with reconnect. A read failure closes the capture and reopens it
/// after an exponential backoff (1 s .. 15 s); the caller only ever sees frames or cancellation.
///
/// RU: Живой источник (RTSP или индекс камеры) с переподключением. Ошибка чтения закрывает захват и открывает его
/// заново после экспоненциальной паузы (1 с .. 15 с); вызывающая сторона видит только кадры или отмену.
/// </summary>
internal sealed class LiveCaptureFrameSource : IFrameSource
{
    static readonly TimeSpan MinBackoff = TimeSpan.FromSeconds(1);
    static readonly TimeSpan MaxBackoff = TimeSpan.FromSeconds(15);

    readonly string? url;
    readonly int? cameraIndex;
    readonly VideoAccelerationType acceleration;
    readonly Mat reusableFrame = new();
    VideoCapture? capture;
    TimeSpan backoff = MinBackoff;

    LiveCaptureFrameSource(string? url, int? cameraIndex, string description, VideoAccelerationType acceleration = VideoAccelerationType.None)
    {
        this.acceleration = acceleration;
        this.url = url;
        this.cameraIndex = cameraIndex;
        Description = description;
    }

    public string Description { get; }

    public double Fps => 0;

    public int ReconnectCount { get; private set; }

    public static LiveCaptureFrameSource FromRtsp(string url, string transport, VideoAccelerationType acceleration)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);

        // OpenCV's FFmpeg backend reads capture options from this process-wide variable. TCP avoids UDP packet loss
        // artifacts on 4K HEVC streams; this mirrors "-rtsp_transport tcp" used by the recorder scripts.
        if(!string.IsNullOrWhiteSpace(transport))
            Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS", $"rtsp_transport;{transport}");

        return new LiveCaptureFrameSource(url, null, $"rtsp {MaskCredentials(url)}", acceleration);
    }

    public static LiveCaptureFrameSource FromCameraIndex(int index) =>
        new(null, index, $"camera #{index}");

    public ISourceFrame? ReadNext(CancellationToken cancellationToken)
    {
        while(!cancellationToken.IsCancellationRequested)
        {
            if(capture is null && !TryOpen(cancellationToken))
                continue;

            if(capture!.Read(reusableFrame) && !reusableFrame.Empty())
            {
                backoff = MinBackoff;
                return new MatSourceFrame(reusableFrame.Clone());
            }

            CloseCurrent();
        }

        return null;
    }

    bool TryOpen(CancellationToken cancellationToken)
    {
        VideoCapture? candidate;
        if(url is not null)
        {
            candidate = VideoCaptureFactory.TryOpen(url, acceleration);
        }
        else
        {
            candidate = new VideoCapture(cameraIndex!.Value);
            if(!candidate.IsOpened())
            {
                candidate.Dispose();
                candidate = null;
            }
        }

        if(candidate is not null)
        {
            capture = candidate;
            return true;
        }

        ReconnectCount++;
        cancellationToken.WaitHandle.WaitOne(backoff);
        backoff = TimeSpan.FromTicks(Math.Min(backoff.Ticks * 2, MaxBackoff.Ticks));
        return false;
    }

    void CloseCurrent()
    {
        capture?.Dispose();
        capture = null;
    }

    static string MaskCredentials(string url)
    {
        int schemeEnd = url.IndexOf("://", StringComparison.Ordinal);
        int at = url.IndexOf('@');
        return schemeEnd >= 0 && at > schemeEnd
            ? string.Concat(url.AsSpan(0, schemeEnd + 3), "***", url.AsSpan(at))
            : url;
    }

    public void Dispose()
    {
        CloseCurrent();
        reusableFrame.Dispose();
    }
}
