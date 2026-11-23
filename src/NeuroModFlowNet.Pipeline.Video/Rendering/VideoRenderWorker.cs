using System.Diagnostics;
using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class VideoRenderWorker : IDisposable
{
    readonly VideoStreamProfile profile;
    readonly string rendererLibrary;
    readonly IDirect2DDrawer drawer;
    readonly SemaphoreSlim gpuWork;
    readonly SemaphoreSlim deviceWork;
    readonly Action<NvencSurface, long, long> publish;
    readonly Action<VideoOutputEvent>? notify;
    readonly object gate = new();
    readonly CancellationTokenSource stop = new();
    readonly Task worker;
    D3D11VideoRenderer? renderer;
    D3D11Nv12SurfacePool? surfaces;
    VideoInputFrame? pending;
    long latestSequence = long.MinValue;
    bool accepting = true;

    public VideoRenderWorker(VideoStreamProfile profile, string rendererLibrary, IDirect2DDrawer drawer, SemaphoreSlim gpuWork,
        SemaphoreSlim deviceWork, Action<NvencSurface, long, long> publish, Action<VideoOutputEvent>? notify)
    {
        this.profile = profile;
        this.rendererLibrary = rendererLibrary;
        this.drawer = drawer;
        this.gpuWork = gpuWork;
        this.deviceWork = deviceWork;
        this.publish = publish;
        this.notify = notify;
        worker = Task.Run(RunAsync);
    }

    #region Input ownership

    public bool IsAccepting
    {
        get { lock(gate) return accepting; }
    }

    public bool Post(VideoInputFrame frame)
    {
        VideoInputFrame? release;
        bool accepted;
        lock(gate)
        {
            accepted = accepting && frame.Sequence > latestSequence;
            if(accepted)
            {
                latestSequence = frame.Sequence;
                release = pending;
                pending = frame;
            }
            else
                release = frame;
        }
        release?.Dispose();
        return accepted;
    }

    bool HasPending()
    {
        lock(gate) return pending is not null;
    }

    VideoInputFrame? TakePending()
    {
        lock(gate)
        {
            VideoInputFrame? input = pending;
            pending = null;
            return input;
        }
    }

    void CloseInput()
    {
        lock(gate)
            accepting = false;
        TakePending()?.Dispose();
    }

    #endregion

    #region Render execution

    async Task RunAsync()
    {
        CancellationToken cancellationToken = stop.Token;
        try
        {
            await gpuWork.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                renderer = new(profile, rendererLibrary);
                surfaces = new(renderer.Device, profile.OutputWidth, profile.OutputHeight);
            }
            finally { gpuWork.Release(); }

            var program = new VideoRenderProgram(renderer, surfaces, profile.DrawerName ?? RectangleVideoDrawer.RegisterKey, drawer);
            Report("ready");
            using var clock = new PeriodicTimer(TimeSpan.FromSeconds(1d / profile.FramesPerSecond));
            while(await clock.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
            {
                if(!HasPending())
                    continue;

                await deviceWork.WaitAsync(cancellationToken).ConfigureAwait(false);
                try
                {
                    await gpuWork.WaitAsync(cancellationToken).ConfigureAwait(false);
                    try
                    {
                        using VideoInputFrame? input = TakePending();
                        if(input is null || Stopwatch.GetElapsedTime(input.SubmittedAt).TotalMilliseconds > profile.MaximumFrameAgeMilliseconds)
                            continue;

                        NvencSurface prepared = await program.ExecuteAsync(profile.StreamId, input, cancellationToken).ConfigureAwait(false);
                        publish(prepared, input.Sequence, input.SubmittedAt);
                    }
                    finally { gpuWork.Release(); }
                }
                finally { deviceWork.Release(); }
            }
        }
        catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested) { }
        catch(Exception exception) { Report("failed", exception.GetBaseException().Message); }
        finally { CloseInput(); }
    }

    void Report(string state, string? detail = null)
    {
        try { notify?.Invoke(new(profile.StreamId, "render", state, detail)); }
        catch(Exception exception) { Trace.TraceError(exception.ToString()); }
    }

    public async Task StopAsync()
    {
        CloseInput();
        await stop.CancelAsync().ConfigureAwait(false);
        await worker.ConfigureAwait(false);
    }

    public void Dispose()
    {
        surfaces?.Dispose();
        renderer?.Dispose();
        stop.Dispose();
        Report("closed");
    }

    #endregion
}
