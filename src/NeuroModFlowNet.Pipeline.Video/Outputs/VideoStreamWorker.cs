using System.Diagnostics;
using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class VideoStreamWorker : IAsyncDisposable
{
    readonly VideoStreamProfile profile;
    readonly SemaphoreSlim gpuWork;
    readonly SemaphoreSlim deviceWork = new(1, 1);
    readonly Action<VideoOutputEvent>? notify;
    readonly object gate = new();
    readonly CancellationTokenSource stop = new();
    readonly VideoRenderWorker renderer;
    readonly Task worker;
    NvencSurface? ready;
    long readyTimestamp;
    long latestSequence = long.MinValue;
    volatile bool accepting = true;

    public VideoStreamWorker(VideoStreamProfile profile, string rendererLibrary, SemaphoreSlim gpuWork, Action<VideoOutputEvent>? notify)
    {
        this.profile = profile;
        this.gpuWork = gpuWork;
        this.notify = notify;
        renderer = new(profile, rendererLibrary, gpuWork, deviceWork, PublishRendered, notify);
        worker = Task.Run(RunAsync);
    }

    public bool IsAccepting => accepting && renderer.IsAccepting;

    public bool Post(VideoInputFrame frame)
    {
        if(accepting)
            return renderer.Post(frame);
        frame.Dispose();
        return false;
    }

    #region Prepared frames from the render VM

    void PublishRendered(NvencSurface prepared, long sequence, long timestamp)
    {
        NvencSurface? release;
        lock(gate)
        {
            if(!accepting || sequence <= latestSequence)
                release = prepared;
            else
            {
                latestSequence = sequence;
                release = ready;
                ready = prepared;
                readyTimestamp = timestamp;
            }
        }
        release?.Dispose();
    }

    (NvencSurface? Surface, long Timestamp) TakeReady()
    {
        lock(gate)
        {
            NvencSurface? surface = ready;
            ready = null;
            return (surface, readyTimestamp);
        }
    }

    bool HasReady()
    {
        lock(gate) return ready is not null;
    }

    void CloseInput()
    {
        lock(gate)
            accepting = false;
        TakeReady().Surface?.Dispose();
    }

    #endregion

    #region Encoding and output clock

    async Task RunAsync()
    {
        D3D11NvencEncoder? encoder = null;
        NvencSurface? last = null;
        EncodedVideoOutput? output = null;
        long lastFrameTime = 0;
        long origin = 0;
        long lastTick = -1;
        try
        {
            using var clock = new PeriodicTimer(TimeSpan.FromSeconds(1d / profile.FramesPerSecond));
            while(await clock.WaitForNextTickAsync(stop.Token).ConfigureAwait(false))
            {
                if(!HasReady() && (last is null || !profile.RepeatLastFrame || IsExpired(lastFrameTime)))
                    continue;

                await deviceWork.WaitAsync(stop.Token).ConfigureAwait(false);
                try
                {
                    await gpuWork.WaitAsync(stop.Token).ConfigureAwait(false);
                    try
                    {
                        var (completed, capturedAt) = TakeReady();
                        bool newFrame = completed is not null && !IsExpired(capturedAt);
                        if(newFrame)
                        {
                            last?.Dispose();
                            last = completed;
                            lastFrameTime = capturedAt;
                        }
                        else
                            completed?.Dispose();

                        if(last is null || IsExpired(lastFrameTime) || (!profile.RepeatLastFrame && !newFrame))
                            continue;

                        if(encoder is null)
                        {
                            encoder = new(profile, last);
                            output = new(encoder.Description, profile.Outputs, notify);
                            origin = Stopwatch.GetTimestamp();
                            Report("ready");
                        }

                        long tick = Math.Max(lastTick + 1, (long)(Stopwatch.GetElapsedTime(origin).TotalSeconds * profile.FramesPerSecond));
                        encoder.Encode(last, tick, output!.Publish);
                        lastTick = tick;
                    }
                    finally { gpuWork.Release(); }
                }
                finally { deviceWork.Release(); }
            }
        }
        catch(OperationCanceledException) when(stop.IsCancellationRequested) { }
        catch(Exception exception) { Report("failed", exception.GetBaseException().Message); }
        finally
        {
            CloseInput();
            try
            {
                await renderer.StopAsync().ConfigureAwait(false);
                if(encoder is not null && output is not null)
                    encoder.Flush(output.Publish);
            }
            catch(Exception exception) { Report("flush-failed", exception.GetBaseException().Message); }
            finally
            {
                last?.Dispose();
                encoder?.Dispose();
                // NVENC must finish using the renderer's CUDA and D3D11 resources before teardown.
                renderer.Dispose();
                if(output is not null)
                    await output.DisposeAsync().ConfigureAwait(false);
                Report("closed");
            }
        }
    }

    bool IsExpired(long timestamp) => Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds > profile.MaximumFrameAgeMilliseconds;

    void Report(string state, string? detail = null)
    {
        try { notify?.Invoke(new(profile.StreamId, "encoder", state, detail)); }
        catch(Exception exception) { Trace.TraceError(exception.ToString()); }
    }

    public async ValueTask DisposeAsync()
    {
        CloseInput();
        await stop.CancelAsync().ConfigureAwait(false);
        try { await worker.ConfigureAwait(false); }
        finally
        {
            stop.Dispose();
            deviceWork.Dispose();
        }
    }

    #endregion
}
