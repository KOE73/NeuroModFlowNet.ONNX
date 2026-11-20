namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Lease-gated catalog of names and instruction boundaries. It retains metadata only, never register payloads or contexts.
///
/// RU: Каталог имён и границ инструкций с разрешением по времени. Хранит только метаданные, без значений регистров и контекстов.
/// </summary>
internal sealed class RegisterCatalog(Func<VmRunContext, string, DebugVariable> describe, TimeProvider? timeProvider = null, int capacity = 512)
{
    readonly TimeProvider clock = timeProvider ?? TimeProvider.System;
    readonly object syncRoot = new();
    readonly Dictionary<string, Entry> entries = new(StringComparer.Ordinal);
    readonly Dictionary<(string, int, string), RegisterCatalogBoundary> boundaries = [];
    long leaseUntil, epoch, observedEpoch, visit, runId, revision;
    long? frameId;
    bool runOpen, truncated;
    RegisterCatalogBoundary? previousBoundary;

    public RegisterCatalogCost Cost { get; } = new();
    public bool Enabled
    {
        get
        {
            long until = Volatile.Read(ref leaseUntil);
            if(until == 0) return false;
            if(clock.GetTimestamp() < until) return true;
            Interlocked.CompareExchange(ref leaseUntil, 0, until);
            return false;
        }
    }

    #region Program boundaries and browser lease

    public void Register(DebugInstruction instruction)
    {
        lock(syncRoot)
            foreach(string phase in new[] { "before", "after" })
                boundaries.Add((instruction.Program, instruction.Index, phase), new(boundaries.Count, instruction.Program,
                    instruction.Index, phase, instruction.Name, phase == "after" ? instruction.Writes : []));
    }

    public RegisterCatalogBoundary Boundary(string program, int index, string phase)
    {
        lock(syncRoot) return boundaries[(program, index, phase)];
    }

    public void Renew()
    {
        long now = clock.GetTimestamp();
        lock(syncRoot)
        {
            if(now >= leaseUntil) epoch++;
            Volatile.Write(ref leaseUntil, now + 5 * clock.TimestampFrequency);
        }
    }

    #endregion

    #region VM thread only

    public void Visit(VmRunContext context, RegisterCatalogBoundary boundary)
    {
        // No enumeration, metadata calls or allocations without an active browser lease.
        if(!Enabled) return;
        if(Cost.Requested && Cost.Begin() is > 0 and var started)
        {
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            try { VisitEnabled(context, boundary); }
            finally { Cost.Add(started, allocated); }
        }
        else VisitEnabled(context, boundary);
    }

    void VisitEnabled(VmRunContext context, RegisterCatalogBoundary boundary)
    {
        long now = clock.GetTimestamp();
        lock(syncRoot)
        {
            bool contiguous = observedEpoch == epoch && runId == context.Identity.RunId && runOpen;
            bool followsBefore = contiguous && boundary.Phase == "after" && previousBoundary is { Phase: "before" } previous &&
                previous.Program == boundary.Program && previous.Index == boundary.Index;
            observedEpoch = epoch;
            runId = context.Identity.RunId;
            frameId = context.Identity.SourceFrameId;
            runOpen = true;
            visit++;
            // VmRunContext exposes its Dictionary keys as an interface. Preserve the struct enumerator on that path.
            if(context.Keys is Dictionary<string, object?>.KeyCollection keys)
                foreach(string key in keys) ObserveKey(context, key, boundary, now, followsBefore);
            else
                foreach(string key in context.Keys) ObserveKey(context, key, boundary, now, followsBefore);
            if(contiguous)
                foreach(Entry entry in entries.Values)
                    if(entry.SeenVisit == visit - 1 && entry.RunId == runId)
                    {
                        entry.RemovedBoundary = boundary;
                        entry.RemovedRun = runId;
                        entry.RemovedFrame = frameId;
                        entry.RemovedVisit = visit;
                    }
            previousBoundary = boundary;
            revision++;
        }
    }

    void ObserveKey(VmRunContext context, string key, RegisterCatalogBoundary boundary, long now, bool followsBefore)
    {
        if(!entries.TryGetValue(key, out Entry? entry))
        {
            if(entries.Count >= capacity) { truncated = true; return; }
            entries.Add(key, entry = new(entries.Count + 1));
        }
        // A declared write or late attachment is not evidence of creation. Require absence before and presence after this op.
        if(entry.CreatedAt is null && followsBefore && entry.SeenVisit != visit - 1)
            entry.CreatedAt = Location(boundary, runId, frameId);
        if(!entry.Points.TryGetValue(boundary.Id, out Point? point))
        {
            point = new(boundary, Metadata(context, key), now);
            entry.Points.Add(boundary.Id, point);
        }
        else if(now - point.Refreshed >= clock.TimestampFrequency)
        {
            point.Variable = Metadata(context, key);
            point.Refreshed = now;
        }
        entry.Variable = point.Variable;
        entry.SeenVisit = visit; entry.RunId = runId; entry.FrameId = frameId; entry.LastBoundary = boundary;
    }

    public void RunCompleted(long completedRun)
    {
        if(!Enabled) return;
        lock(syncRoot) if(runId == completedRun) { runOpen = false; revision++; }
    }

    public void RememberPause(DebugVariable[] variables, DebugLocation location)
    {
        lock(syncRoot)
        {
            if(!boundaries.TryGetValue((location.Program, location.Index, location.Phase), out var boundary)) return;
            runId = location.RunId; frameId = location.FrameId; runOpen = true;
            observedEpoch = -1; visit++;
            foreach(DebugVariable variable in variables)
            {
                if(!entries.TryGetValue(variable.Key, out Entry? entry))
                {
                    if(entries.Count >= capacity) { truncated = true; continue; }
                    entries.Add(variable.Key, entry = new(entries.Count + 1));
                }
                DebugVariable metadata = variable with { Value = null, Points = null, Presence = null, LastSeen = null, RemovedAt = null };
                entry.Points[boundary.Id] = new(boundary, metadata, clock.GetTimestamp());
                entry.Variable = metadata;
                entry.LastBoundary = boundary; entry.RunId = runId; entry.FrameId = frameId; entry.SeenVisit = visit;
            }
            revision++;
        }
    }

    #endregion

    #region Immutable HTTP metadata

    public DebugRegisterCatalogSnapshot Snapshot()
    {
        lock(syncRoot)
            return new(revision, Enabled, truncated, entries.Values.Select(entry => entry.Variable! with
            {
                AdditionOrder = entry.AdditionOrder,
                CreatedAt = entry.CreatedAt,
                Presence = !Enabled ? "catalog-idle" : entry.RunId != runId ? "previous-run" :
                    entry.RemovedVisit > entry.SeenVisit ? "removed" : !runOpen ? "run-ended" : "seen",
                LastSeen = Location(entry.LastBoundary!, entry.RunId, entry.FrameId),
                RemovedAt = entry.RemovedBoundary is { } removed ? Location(removed, entry.RemovedRun, entry.RemovedFrame) : null,
                Points = entry.Points.Values.Select(point => new DebugRegisterPoint(point.Boundary.Program, point.Boundary.Index,
                    point.Boundary.Phase, point.Boundary.Name, point.Boundary.Writes.Contains(point.Variable.Key), point.Variable)).ToArray()
            }).OrderBy(variable => variable.Key, StringComparer.Ordinal).ToArray());
    }

    public DebugWatchTarget Resolve(DebugWatchSelection selection)
    {
        lock(syncRoot)
        {
            if(!entries.TryGetValue(selection.Key, out Entry? entry)) throw new ArgumentException("Register has not been observed yet.");
            bool explicitPoint = selection.Program is not null || selection.Index is not null || selection.Phase is not null;
            Point? point = entry.Points.Values.LastOrDefault(point => point.Boundary.Writes.Contains(selection.Key)) ?? entry.Points.Values.FirstOrDefault();
            RegisterCatalogBoundary? boundary = point?.Boundary;
            if(explicitPoint)
            {
                if(selection.Program is null || selection.Index is null || selection.Phase is null ||
                    !boundaries.TryGetValue((selection.Program, selection.Index.Value, selection.Phase), out boundary))
                    throw new ArgumentException("Choose an installed program, instruction and before/after boundary.");
                point = entry.Points.GetValueOrDefault(boundary.Id);
            }
            if(boundary is null) throw new ArgumentException("Register has not been observed yet.");
            // A common capture point may precede creation or follow removal. Retain the selection; never substitute another step.
            DebugVariable metadata = point?.Variable ?? entry.Variable!;
            if(selection.Format == "h264")
            {
                string[] dimensions = metadata.Shape?.Split('×') ?? [];
                if(metadata.Memory != "Cuda:0" || metadata.ElementType != "UInt8" || dimensions.Length != 4 ||
                    dimensions[0] != "1" || dimensions[3] != "3" ||
                    !int.TryParse(dimensions[1], out int height) || !int.TryParse(dimensions[2], out int width) ||
                    height <= 0 || width <= 0 || height % 2 != 0 || width % 2 != 0)
                    throw new NotSupportedException("H.264 BGR U8 NHWC requires a CUDA:0 tensor [1,H,W,3] with even dimensions.");
            }
            if(selection.Format is "png" or "h264" ? !metadata.CanPreviewImage : selection.Format != "json" || !metadata.CanInspectValue)
                throw new ArgumentException("This register does not support the selected format at that instruction boundary.");
            return new(selection.Key, selection.Format, boundary.Program, boundary.Index, boundary.Phase);
        }
    }

    static DebugLocation Location(RegisterCatalogBoundary boundary, long run, long? frame) =>
        new(boundary.Program, boundary.Index, boundary.Name, boundary.Phase, run, frame);

    DebugVariable Metadata(VmRunContext context, string key)
    {
        try { return describe(context, key) with { Value = null }; }
        catch(Exception exception) when(exception is not OutOfMemoryException)
        {
            context.TryGetObject(key, out object? value);
            return new(key, value?.GetType().FullName ?? "null", "Unknown", null, null, null, MetadataError: exception.GetBaseException().Message);
        }
    }

    #endregion

    sealed class Entry(int additionOrder)
    {
        public int AdditionOrder { get; } = additionOrder;
        public Dictionary<int, Point> Points { get; } = [];
        public DebugVariable? Variable { get; set; }
        public DebugLocation? CreatedAt { get; set; }
        public long SeenVisit, RunId, RemovedRun, RemovedVisit;
        public long? FrameId, RemovedFrame;
        public RegisterCatalogBoundary? LastBoundary, RemovedBoundary;
    }

    sealed class Point(RegisterCatalogBoundary boundary, DebugVariable variable, long refreshed)
    {
        public RegisterCatalogBoundary Boundary { get; } = boundary;
        public DebugVariable Variable { get; set; } = variable;
        public long Refreshed { get; set; } = refreshed;
    }
}
