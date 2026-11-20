using System.Diagnostics;
using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class VideoStreamWorker : IAsyncDisposable
{
    readonly VideoStreamProfile profile;
    readonly string rendererLibrary;
    readonly SemaphoreSlim gpuWork;
    readonly Action<VideoOutputEvent>? notify;
    readonly object gate = new();
    readonly CancellationTokenSource stop = new();
    readonly Task worker;
    VideoInputFrame? pending;
    long latestSequence = long.MinValue;
    bool accepting = true;

    public VideoStreamWorker(VideoStreamProfile profile, string rendererLibrary, SemaphoreSlim gpuWork, Action<VideoOutputEvent>? notify)
    {
        this.profile = profile;
        this.rendererLibrary = rendererLibrary;
        this.gpuWork = gpuWork;
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

    VideoInputFrame? TakePending()
    {
        lock(gate)
        {
            VideoInputFrame? frame = pending;
            pending = null;
            return frame;
        }
    }

    bool HasPending()
    {
        lock(gate) return pending is not null;
    }

    void CloseInput()
    {
        lock(gate)
            accepting = false;
        TakePending()?.Dispose();
    }

    #endregion

    #region Output clock and GPU work

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
                if(!HasPending() && (last is null || !profile.RepeatLastFrame || IsExpired(lastFrameTime)))
                    continue;

                await gpuWork.WaitAsync(stop.Token).ConfigureAwait(false);
                try
                {
                    using VideoInputFrame? input = TakePending();
                    bool renderedNewFrame = false;
                    if(input is not null && !IsExpired(input.SubmittedAt))
                    {
                        if(encoder is null)
                        {
                            encoder = new(profile, rendererLibrary);
                            output = new(encoder.Description, profile.Outputs, notify);
                            origin = Stopwatch.GetTimestamp();
                            Report("ready");
                        }
                        NvencSurface rendered = encoder.Render(input);
                        last?.Dispose();
                        last = rendered;
                        lastFrameTime = input.SubmittedAt;
                        renderedNewFrame = true;
                    }

                    if(last is null || IsExpired(lastFrameTime))
                        continue;
                    if(!profile.RepeatLastFrame && !renderedNewFrame)
                        continue;

                    long tick = Math.Max(lastTick + 1, (long)(Stopwatch.GetElapsedTime(origin).TotalSeconds * profile.FramesPerSecond));
                    encoder!.Encode(last, tick, output!.Publish);
                    lastTick = tick;
                }
                finally { gpuWork.Release(); }
            }
        }
        catch(OperationCanceledException) when(stop.IsCancellationRequested) { }
        catch(Exception exception) { Report("failed", exception.GetBaseException().Message); }
        finally
        {
            CloseInput();
            try
            {
                if(encoder is not null && output is not null)
                    encoder.Flush(output.Publish);
            }
            catch(Exception exception) { Report("flush-failed", exception.GetBaseException().Message); }
            finally
            {
                last?.Dispose();
                encoder?.Dispose();
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
        finally { stop.Dispose(); }
    }

    #endregion
}
