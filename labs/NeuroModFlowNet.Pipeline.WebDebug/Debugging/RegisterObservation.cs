using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Bounded pull-driven observation. A slot is reserved before capture; only owned CPU data reaches an encoder.
///
/// RU: Ограниченное наблюдение по запросу браузера. Слот резервируется до снимка; кодировщик получает только владеющие CPU-данные.
/// </summary>
internal sealed class RegisterObservation(Func<VmRunContext, DebugInspectionRequest, CancellationToken, ValueTask<CapturedRegister>> capture) : IDisposable
{
    readonly object syncRoot = new();
    readonly HashSet<Task> workers = [];
    WatchState[] watches = [];
    DebugWatchConfiguration configuration = new(false, 3, 640, []);
    long generation, sequence, leaseExpires;
    int inFlight;
    bool enabled, disposed;

    public bool Enabled => Volatile.Read(ref enabled);

    #region Configuration and delivery

    public DebugWatchSnapshot Configure(DebugWatchConfiguration value, DebugWatchTarget[] targets)
    {
        if(!double.IsFinite(value.FramesPerSecond) || value.FramesPerSecond is < 0.2 or > 15)
            throw new ArgumentException("Observation frequency must be between 0.2 and 15 FPS per register.");
        if(value.MaxEdge is < 100 or > 2000) throw new ArgumentException("Observation image size must be between 100 and 2000 pixels.");
        if(targets.Length > 8 || targets.Select(target => target.Key).Distinct(StringComparer.Ordinal).Count() != targets.Length)
            throw new ArgumentException("Select up to eight different registers.");
        lock(syncRoot)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Volatile.Write(ref enabled, false);
            configuration = value;
            generation++;
            Volatile.Write(ref watches, targets.Select(target => new WatchState(target, generation)).ToArray());
            leaseExpires = 0;
            return Snapshot(false);
        }
    }

    public DebugWatchSnapshot Snapshot(bool includeData = false)
    {
        lock(syncRoot)
            return new(generation, configuration, watches.Select(watch => new DebugWatchItem(watch.Target, watch.Captured, watch.Skipped,
                includeData ? watch.Frame : watch.Frame is { } frame ? frame with { Data = null } : null)).ToArray());
    }

    public DebugWatchSnapshot Poll(DebugWatchAcknowledgement acknowledgement)
    {
        if(acknowledgement.After is null || acknowledgement.After.Count > 8 || acknowledgement.After.Values.Any(value => value < 0))
            throw new ArgumentException("Invalid observation acknowledgements.");
        lock(syncRoot)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if(acknowledgement.Generation != generation) return Snapshot();
            leaseExpires = Stopwatch.GetTimestamp() + Stopwatch.Frequency * 5;
            foreach(WatchState watch in watches)
            {
                long after = acknowledgement.After.GetValueOrDefault(watch.Target.Key);
                if(after > (watch.Frame?.Sequence ?? 0)) throw new ArgumentException("Cannot acknowledge an unseen observation frame.");
                if(after == (watch.Frame?.Sequence ?? 0) && !watch.Busy) watch.Requested = true;
            }
            Volatile.Write(ref enabled, configuration.Enabled && watches.Length > 0);
            return new(generation, configuration, watches.Select(watch => new DebugWatchItem(watch.Target, watch.Captured, watch.Skipped,
                watch.Frame is { } frame && frame.Sequence > acknowledgement.After.GetValueOrDefault(watch.Target.Key) ? frame : null)).ToArray());
        }
    }

    #endregion

    #region VM capture and independent encoding

    public ValueTask CaptureAsync(VmRunContext context, string program, int index, string phase, string name, CancellationToken cancellationToken) =>
        Enabled ? CaptureEnabledAsync(context, program, index, phase, name, cancellationToken) : ValueTask.CompletedTask;

    async ValueTask CaptureEnabledAsync(VmRunContext context, string program, int index, string phase, string name, CancellationToken cancellationToken)
    {
        if(!Enabled) return;
        foreach(WatchState watch in Volatile.Read(ref watches))
        {
            if(watch.Target.Program != program || watch.Target.Index != index || watch.Target.Phase != phase) continue;
            int maxEdge;
            lock(syncRoot)
            {
                if(!Enabled || watch.Generation != generation) continue;
                long now = Stopwatch.GetTimestamp();
                if(now > leaseExpires) { Volatile.Write(ref enabled, false); return; }
                if(watch.Busy || !watch.Requested || inFlight >= 8 || now < watch.NextCapture)
                {
                    watch.Skipped++;
                    continue;
                }
                watch.Busy = true;
                inFlight++;
                watch.Requested = false;
                watch.NextCapture = now + (long)(Stopwatch.Frequency / configuration.FramesPerSecond);
                maxEdge = configuration.MaxEdge;
            }

            var location = new DebugLocation(program, index, name, phase, context.Identity.RunId, context.Identity.SourceFrameId);
            DateTimeOffset capturedAt = DateTimeOffset.UtcNow;
            CapturedRegister? captured = null;
            try
            {
                // The reservation above is the only path to GPU preparation/download or value serialization.
                captured = await capture(context, new(watch.Target.Key, 0, watch.Target.Format, maxEdge), cancellationToken).ConfigureAwait(false);
                lock(syncRoot)
                {
                    if(disposed || watch.Generation != generation) { inFlight--; captured.Dispose(); captured = null; continue; }
                    CapturedRegister owned = captured;
                    Task worker = Task.Run(() => Encode(watch, owned, location, capturedAt));
                    captured = null;
                    workers.Add(worker);
                    _ = worker.ContinueWith(completed => { lock(syncRoot) workers.Remove(completed); }, CancellationToken.None,
                        TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
                }
            }
            catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested)
            {
                lock(syncRoot) { watch.Busy = false; inFlight--; }
                throw;
            }
            catch(Exception exception) { Publish(watch, location, capturedAt, null, exception.GetBaseException().Message); }
            finally { captured?.Dispose(); }
        }
    }

    void Encode(WatchState watch, CapturedRegister captured, DebugLocation location, DateTimeOffset capturedAt)
    {
        DebugInspectionResult? result = null;
        string? error = null;
        try
        {
            using(captured) result = captured.Encode();
            if(result.Data.Length > 16 * 1024 * 1024) throw new InvalidOperationException("Observation exceeds the 16 MiB sample limit.");
        }
        catch(Exception exception) { result = null; error = exception.GetBaseException().Message; }
        Publish(watch, location, capturedAt, result, error);
    }

    void Publish(WatchState watch, DebugLocation location, DateTimeOffset capturedAt, DebugInspectionResult? result, string? error)
    {
        lock(syncRoot)
        {
            watch.Busy = false;
            inFlight--;
            if(disposed || watch.Generation != generation) return;
            watch.Captured++;
            watch.Frame = new(++sequence, location, capturedAt, result?.ContentType, result?.Data, error);
        }
    }

    #endregion

    public void Dispose()
    {
        Task[] pending;
        lock(syncRoot)
        {
            if(disposed) return;
            disposed = true;
            Volatile.Write(ref enabled, false);
            generation++;
            watches = [];
            pending = workers.ToArray();
        }
        // Encoding owns detached CPU images; drain it before unloading the debug host's native libraries.
        Task.WhenAll(pending).GetAwaiter().GetResult();
    }

    sealed class WatchState(DebugWatchTarget target, long generation)
    {
        public DebugWatchTarget Target { get; } = target;
        public long Generation { get; } = generation;
        public bool Requested { get; set; }
        public bool Busy { get; set; }
        public long NextCapture { get; set; }
        public long Captured { get; set; }
        public long Skipped { get; set; }
        public DebugWatchFrame? Frame { get; set; }
    }
}
