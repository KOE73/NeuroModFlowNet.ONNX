using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Independent viewer subscriptions share bounded captures of identical register/size targets.
///
/// RU: Независимые подписки окон разделяют ограниченные снимки одинаковых регистров и размеров.
/// </summary>
internal sealed class RegisterObservation(Func<VmRunContext, DebugInspectionRequest, CancellationToken, ValueTask<CapturedRegister>> capture) : IDisposable
{
    readonly object syncRoot = new();
    readonly Dictionary<string, Viewer> viewers = new(StringComparer.Ordinal);
    readonly HashSet<Task> workers = [];
    Target[] targets = [];
    long generation, sequence;
    int inFlight;
    bool enabled, disposed;
    public bool Enabled => Volatile.Read(ref enabled);

    #region Configuration and delivery

    Viewer GetViewer(string id)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if(string.IsNullOrEmpty(id) || id.Length > 80 || id.Any(character => !char.IsAsciiLetterOrDigit(character) && character is not '-' and not '_'))
            throw new ArgumentException("Invalid viewer identifier.");
        if(viewers.TryGetValue(id, out Viewer? viewer)) return viewer;
        foreach(string expired in viewers.Where(pair => pair.Key != "main" && Stopwatch.GetElapsedTime(pair.Value.LastSeen).TotalMinutes > 10).Select(pair => pair.Key).ToArray()) viewers.Remove(expired);
        if(viewers.Count >= 8) throw new InvalidOperationException("Up to eight observation windows may be connected.");
        viewer = new(); viewers.Add(id, viewer); return viewer;
    }

    public DebugWatchSnapshot Configure(DebugWatchConfiguration value, DebugWatchTarget[] selections, string viewerId = "main")
    {
        if(!double.IsFinite(value.FramesPerSecond) || value.FramesPerSecond is < 0.2 or > 15) throw new ArgumentException("Observation frequency must be between 0.2 and 15 FPS per register.");
        if(!PreviewImageSize.IsValid(value.MaxEdge)) throw new ArgumentException("Use 0 for original size or 100–8192 pixels.");
        if(selections.Length > 8 || selections.Select(target => target.Key).Distinct(StringComparer.Ordinal).Count() != selections.Length) throw new ArgumentException("Select up to eight different registers.");
        lock(syncRoot)
        {
            Viewer viewer = GetViewer(viewerId);
            var variants = viewers.Values.Where(item => item != viewer && item.Configuration.Enabled)
                .SelectMany(item => item.Watches.Select(watch => (watch.Target.Selection, watch.Target.MaxEdge)))
                .Concat(value.Enabled ? selections.Select(selection => (selection, value.MaxEdge)) : []).Distinct().ToArray();
            if(variants.Length > 16) throw new InvalidOperationException("Up to sixteen distinct register/size captures may be observed across windows.");
            viewer.Configuration = value; viewer.Generation = ++generation; viewer.LeaseExpires = 0; viewer.LastSeen = Stopwatch.GetTimestamp();
            viewer.Watches = selections.Select(selection => new Watch(viewer,
                targets.FirstOrDefault(target => target.Selection == selection && target.MaxEdge == value.MaxEdge) ?? new Target(selection, value.MaxEdge))).ToArray();
            Volatile.Write(ref targets, viewers.Values.Where(item => item.Configuration.Enabled).SelectMany(item => item.Watches).Select(watch => watch.Target).Distinct().ToArray());
            RefreshEnabled(Stopwatch.GetTimestamp());
            return Snapshot(false, viewerId);
        }
    }

    public DebugWatchSnapshot Snapshot(bool includeData = false, string viewerId = "main")
    {
        lock(syncRoot)
        {
            Viewer viewer = GetViewer(viewerId);
            return new(viewer.Generation, viewer.Configuration, viewer.Watches.Select(watch => new DebugWatchItem(watch.Target.Selection, watch.Captured, watch.Skipped,
                includeData ? watch.Frame : watch.Frame is { } frame ? frame with { Data = null } : null)).ToArray());
        }
    }

    public DebugWatchSnapshot Poll(DebugWatchAcknowledgement acknowledgement, string viewerId = "main")
    {
        if(acknowledgement.After is null || acknowledgement.After.Count > 8 || acknowledgement.After.Values.Any(value => value < 0)) throw new ArgumentException("Invalid observation acknowledgements.");
        lock(syncRoot)
        {
            Viewer viewer = GetViewer(viewerId);
            if(acknowledgement.Generation != viewer.Generation) return Snapshot(false, viewerId);
            long now = Stopwatch.GetTimestamp(); viewer.LastSeen = now; viewer.LeaseExpires = now + Stopwatch.Frequency * 5;
            foreach(Watch watch in viewer.Watches)
            {
                long after = acknowledgement.After.GetValueOrDefault(watch.Target.Selection.Key);
                if(after > (watch.Frame?.Sequence ?? 0)) throw new ArgumentException("Cannot acknowledge an unseen observation frame.");
                if(after == (watch.Frame?.Sequence ?? 0) && !watch.Busy) watch.Requested = true;
            }
            RefreshEnabled(now);
            return new(viewer.Generation, viewer.Configuration, viewer.Watches.Select(watch => new DebugWatchItem(watch.Target.Selection, watch.Captured, watch.Skipped,
                watch.Frame is { } frame && frame.Sequence > acknowledgement.After.GetValueOrDefault(watch.Target.Selection.Key) ? frame : null)).ToArray());
        }
    }

    void RefreshEnabled(long now) => Volatile.Write(ref enabled, viewers.Values.Any(viewer => viewer.Configuration.Enabled && viewer.Watches.Length > 0 && viewer.LeaseExpires >= now));

    #endregion

    #region VM capture and owned CPU encoding

    public bool NeedsPreviewCapture(string program, int producerIndex, DebugPreviewRegisters registers)
    {
        if(!Enabled) return false;
        lock(syncRoot)
        {
            long now = Stopwatch.GetTimestamp();
            RefreshEnabled(now);
            if(!Enabled || inFlight >= 8) return false;
            // This host has one VM run per camera: only that VM task admits captures, so capacity cannot shrink
            // between this readiness check and the immediately following preview command/capture boundary.
            foreach(Viewer viewer in viewers.Values)
            {
                if(!viewer.Configuration.Enabled || now > viewer.LeaseExpires) continue;
                foreach(Watch watch in viewer.Watches)
                {
                    DebugWatchTarget selection = watch.Target.Selection;
                    if((selection.Key != registers.Prepared && selection.Key != registers.Published) || selection.Program != program ||
                        selection.Index < producerIndex || selection.Index == producerIndex && selection.Phase != "after") continue;
                    if(!watch.Target.Busy && !watch.Busy && watch.Requested && now >= watch.NextCapture) return true;
                }
            }
            return false;
        }
    }

    public ValueTask CaptureAsync(VmRunContext context, string program, int index, string phase, string name, CancellationToken cancellationToken) =>
        Enabled ? CaptureEnabledAsync(context, program, index, phase, name, cancellationToken) : ValueTask.CompletedTask;

    async ValueTask CaptureEnabledAsync(VmRunContext context, string program, int index, string phase, string name, CancellationToken cancellationToken)
    {
        foreach(Target target in Volatile.Read(ref targets))
        {
            if(target.Selection.Format == "h264") continue;
            if(target.Selection.Program != program || target.Selection.Index != index || target.Selection.Phase != phase) continue;
            Watch[] receivers;
            lock(syncRoot)
            {
                long now = Stopwatch.GetTimestamp(); RefreshEnabled(now); if(!Enabled) return;
                var ready = new List<Watch>();
                foreach(Viewer viewer in viewers.Values)
                {
                    if(!viewer.Configuration.Enabled || now > viewer.LeaseExpires) continue;
                    foreach(Watch watch in viewer.Watches)
                    {
                        if(watch.Target != target) continue;
                        if(target.Busy || inFlight >= 8 || watch.Busy || !watch.Requested || now < watch.NextCapture) { watch.Skipped++; continue; }
                        ready.Add(watch);
                    }
                }
                if(ready.Count == 0) continue;
                receivers = ready.ToArray(); target.Busy = true; inFlight++;
                foreach(Watch watch in receivers) { watch.Busy = true; watch.Requested = false; watch.NextCapture = now + (long)(Stopwatch.Frequency / watch.Viewer.Configuration.FramesPerSecond); }
            }
            var location = new DebugLocation(program, index, name, phase, context.Identity.RunId, context.Identity.SourceFrameId);
            DateTimeOffset capturedAt = DateTimeOffset.UtcNow;
            CapturedRegister? captured = null;
            try
            {
                // Admission precedes every serialization, GPU preparation and download, irrespective of viewer count.
                captured = await capture(context, new(target.Selection.Key, 0, target.Selection.Format, target.MaxEdge), cancellationToken).ConfigureAwait(false);
                lock(syncRoot)
                {
                    CapturedRegister owned = captured;
                    Task worker = Task.Run(() => Encode(target, receivers, owned, location, capturedAt)); captured = null;
                    workers.Add(worker);
                    _ = worker.ContinueWith(completed => { lock(syncRoot) workers.Remove(completed); }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
                }
            }
            catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested) { Publish(target, receivers, location, capturedAt, null, null); throw; }
            catch(Exception exception) { Publish(target, receivers, location, capturedAt, null, exception.GetBaseException().Message); }
            finally { captured?.Dispose(); }
        }
    }

    void Encode(Target target, Watch[] receivers, CapturedRegister captured, DebugLocation location, DateTimeOffset capturedAt)
    {
        DebugInspectionResult? result = null; string? error = null;
        try
        {
            using(captured) result = captured.Encode();
            if(result.Data.Length > PreviewImageSize.MaximumEncodedBytes) throw new InvalidOperationException("Observation exceeds the 64 MiB sample limit.");
        }
        catch(Exception exception) { result = null; error = exception.GetBaseException().Message; }
        Publish(target, receivers, location, capturedAt, result, error);
    }

    void Publish(Target target, Watch[] receivers, DebugLocation location, DateTimeOffset capturedAt, DebugInspectionResult? result, string? error)
    {
        lock(syncRoot)
        {
            target.Busy = false; inFlight--; if(disposed) return;
            var frame = new DebugWatchFrame(++sequence, location, capturedAt, result?.ContentType, result?.Data, error);
            foreach(Watch watch in receivers)
            {
                watch.Busy = false;
                if(!watch.Viewer.Watches.Contains(watch)) continue;
                watch.Captured++; watch.Frame = frame;
            }
        }
    }

    #endregion

    public void Dispose()
    {
        Task[] pending;
        lock(syncRoot) { if(disposed) return; disposed = true; Volatile.Write(ref enabled, false); targets = []; viewers.Clear(); pending = workers.ToArray(); }
        Task.WhenAll(pending).GetAwaiter().GetResult();
    }

    sealed class Viewer
    {
        public DebugWatchConfiguration Configuration { get; set; } = new(false, 3, 640, []);
        public long Generation { get; set; }
        public long LeaseExpires { get; set; }
        public long LastSeen { get; set; } = Stopwatch.GetTimestamp();
        public Watch[] Watches { get; set; } = [];
    }
    sealed class Target(DebugWatchTarget selection, int maxEdge)
    {
        public DebugWatchTarget Selection { get; } = selection;
        public int MaxEdge { get; } = maxEdge;
        public bool Busy { get; set; }
    }
    sealed class Watch(Viewer viewer, Target target)
    {
        public Viewer Viewer { get; } = viewer;
        public Target Target { get; } = target;
        public bool Requested { get; set; }
        public bool Busy { get; set; }
        public long NextCapture { get; set; }
        public long Captured { get; set; }
        public long Skipped { get; set; }
        public DebugWatchFrame? Frame { get; set; }
    }
}
