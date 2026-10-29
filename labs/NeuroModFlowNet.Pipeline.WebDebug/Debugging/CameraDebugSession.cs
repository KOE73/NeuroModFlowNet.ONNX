namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Serial pause gate for one camera. Only the stopped VM task may inspect its live registers.
///
/// RU: Последовательный шлюз паузы одной камеры. Только остановленная задача VM читает живые регистры.
/// </summary>
internal sealed class CameraDebugSession(string cameraId, PausedRegisterInspector inspector) : IDisposable
{
    readonly object syncRoot = new();
    readonly SemaphoreSlim wake = new(0);
    readonly Queue<InspectionWork> inspections = new();
    readonly HashSet<DebugBreakpoint> breakpoints = [];
    readonly Dictionary<(string Program, int Index), DebugTiming> timings = [];
    readonly List<DebugInstruction> instructions = [];
    string state = "waiting";
    string? error;
    long revision = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000;
    DebugLocation? location;
    DebugVariable[] variables = [];
    bool attached = true;
    bool pauseBefore = true;
    bool pauseAfter;
    bool resumeRequested;
    bool disposed;
    long? finishFrame;
    DebugBreakpoint? runTo;
    DebugVariable[] knownVariables = [];
    DebugLocation? knownLocation;
    public RegisterObservation Observation { get; } = new(inspector.CaptureAsync);

    public string CameraId { get; } = cameraId;

    #region Program and snapshots

    public VmProgram Decorate(VmProgram program)
    {
        string name = program.Name ?? "program";
        lock(syncRoot)
        {
            instructions.AddRange(program.Instructions.Select((instruction, index) => new DebugInstruction(
                name, index, instruction.Descriptor.Name, instruction.Descriptor.Operation,
                instruction.Descriptor.Reads.Select(item => item.Key).ToArray(),
                instruction.Descriptor.Writes.Select(item => item.Key).ToArray())));
        }
        return new VmProgram(program.Instructions.Select((instruction, index) =>
            instruction is IHasExecutionDevice
                ? (IOp)new DebugDeviceOp(instruction, this, name, index)
                : new DebugOp(instruction, this, name, index)).ToArray(), program.Labels, program.Name);
    }

    public DebugInstruction[] Instructions { get { lock(syncRoot) return instructions.ToArray(); } }

    public DebugSnapshot Snapshot()
    {
        lock(syncRoot)
            return new(state, revision, location, variables, breakpoints.ToArray(), timings.Values.ToArray(), error, runTo);
    }

    public void RecordTiming(string program, int index, string name, double milliseconds)
    {
        lock(syncRoot)
            timings[(program, index)] = new(program, index, name, milliseconds);
    }

    public void RecordWarmup(VmRunContext context, string program, int index, string name)
    {
        lock(syncRoot)
        {
            location = new(program, index, name, "warmup", context.Identity.RunId, context.Identity.SourceFrameId);
            revision++;
        }
    }

    public void RunCompleted(VmRunOutcome outcome)
    {
        lock(syncRoot)
        {
            if(outcome.Status == VmRunStatus.Failed)
            {
                state = "failed";
                runTo = null;
                error = outcome.Exception?.GetBaseException().Message ?? "VM run failed.";
            }
            else if(state != "failed")
                state = attached ? "running" : "detached";
            variables = [];
            revision++;
        }
    }

    public void Fail(Exception exception)
    {
        lock(syncRoot)
        {
            state = "failed";
            runTo = null;
            error = exception.GetBaseException().Message;
            variables = [];
            revision++;
            RejectInspections();
        }
    }

    #endregion

    #region Commands

    public void Command(DebugCommand command)
    {
        lock(syncRoot)
        {
            if(disposed) throw new InvalidOperationException("This debug session has ended.");
            if(command.Revision is { } expected && expected != revision)
                throw new InvalidOperationException("The stop changed. Refresh before sending another command.");

            switch(command.Command)
            {
                case "pause":
                case "attach":
                    attached = true;
                    runTo = null;
                    if(state != "paused") { pauseBefore = true; state = "pausing"; }
                    return;
                case "detach":
                    attached = false;
                    pauseBefore = pauseAfter = false;
                    finishFrame = null;
                    runTo = null;
                    break;
                case "continue":
                    EnsurePaused();
                    pauseBefore = pauseAfter = false;
                    finishFrame = null;
                    runTo = null;
                    break;
                case "step":
                    EnsurePaused();
                    pauseBefore = false;
                    pauseAfter = true;
                    finishFrame = null;
                    runTo = null;
                    break;
                case "next-frame":
                    EnsurePaused();
                    pauseBefore = pauseAfter = false;
                    finishFrame = location!.RunId;
                    runTo = null;
                    break;
                case "run-to":
                    EnsurePaused();
                    if(command.Program is null || command.Index is null ||
                        !instructions.Any(instruction => instruction.Program == command.Program && instruction.Index == command.Index))
                        throw new ArgumentException("Run-to requires an existing program name and instruction index.");
                    if(location is { Phase: "before" } && location.Program == command.Program && location.Index == command.Index)
                        return;
                    runTo = new(command.Program, command.Index.Value);
                    pauseBefore = pauseAfter = false;
                    finishFrame = null;
                    break;
                default:
                    throw new ArgumentException("Unknown command. Use pause, attach, detach, continue, step, next-frame or run-to.");
            }

            resumeRequested = true;
            state = attached ? "running" : "detached";
            variables = [];
            revision++;
            RejectInspections();
            wake.Release();
        }
    }

    public void SetBreakpoints(DebugBreakpoint[] values)
    {
        lock(syncRoot)
        {
            if(values.Any(value => !instructions.Any(instruction => instruction.Program == value.Program && instruction.Index == value.Index)))
                throw new ArgumentException("A breakpoint must identify an existing program and instruction index.");
            breakpoints.Clear();
            breakpoints.UnionWith(values);
        }
    }

    public DebugWatchSnapshot ConfigureObservation(DebugWatchConfiguration configuration)
    {
        lock(syncRoot)
        {
            if(disposed) throw new InvalidOperationException("This debug session has ended.");
            if(configuration.Registers is null || configuration.Registers.Length > 8)
                throw new ArgumentException("Select up to eight registers.");
            DebugWatchTarget[] targets = configuration.Registers.Select(selection =>
            {
                DebugVariable variable = knownVariables.FirstOrDefault(variable => variable.Key == selection.Key)
                    ?? throw new ArgumentException("Select a register available at the last pause.");
                if(selection.Format == "png" ? !variable.CanPreviewImage : selection.Format != "json" || !variable.CanInspectValue)
                    throw new ArgumentException("The selected register does not support this observation format.");
                int stopIndex = instructions.FindIndex(instruction => instruction.Program == knownLocation?.Program && instruction.Index == knownLocation.Index);
                int executed = stopIndex + (knownLocation?.Phase == "after" ? 1 : 0);
                DebugInstruction? writer = instructions.Take(executed).LastOrDefault(instruction => instruction.Writes.Contains(selection.Key));
                DebugInstruction point = writer ?? instructions[0];
                return new DebugWatchTarget(selection.Key, selection.Format, point.Program, point.Index, writer is null ? "before" : "after");
            }).ToArray();
            return Observation.Configure(configuration, targets) with { Variables = knownVariables };
        }
    }

    public DebugWatchSnapshot ObservationSnapshot()
    {
        lock(syncRoot) return Observation.Snapshot() with { Variables = knownVariables };
    }

    public Task<DebugInspectionResult> InspectAsync(DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        lock(syncRoot)
        {
            EnsurePaused();
            if(request.Revision != revision)
                throw new InvalidOperationException("The register belongs to an earlier stop. Refresh and select it again.");
            if(inspections.Count >= 8)
                throw new InvalidOperationException("Too many inspection requests. Wait for the current request.");
            var completion = new TaskCompletionSource<DebugInspectionResult>(TaskCreationOptions.RunContinuationsAsynchronously);
            inspections.Enqueue(new(request, completion, cancellationToken));
            wake.Release();
            return completion.Task.WaitAsync(cancellationToken);
        }
    }

    void EnsurePaused()
    {
        if(state != "paused")
            throw new InvalidOperationException("This action requires a paused camera.");
    }

    void RejectInspections()
    {
        while(inspections.TryDequeue(out InspectionWork? work))
            work.Completion.TrySetException(new InvalidOperationException("The camera resumed; the paused register is no longer available."));
    }

    #endregion

    #region VM task

    public ValueTask BeforeAsync(VmRunContext context, string program, int index, OpDescriptor descriptor, CancellationToken cancellationToken)
    {
        lock(syncRoot)
        {
            if(!attached) return ValueTask.CompletedTask;
            bool stop = pauseBefore || breakpoints.Contains(new(program, index)) ||
                runTo is { } target && target.Program == program && target.Index == index ||
                finishFrame is { } previousRun && previousRun != context.Identity.RunId;
            if(!stop) return ValueTask.CompletedTask;
            pauseBefore = false;
            finishFrame = null;
            runTo = null;
        }
        return PauseAsync(context, program, index, descriptor, "before", cancellationToken);
    }

    public ValueTask AfterAsync(VmRunContext context, string program, int index, OpDescriptor descriptor, CancellationToken cancellationToken)
    {
        lock(syncRoot)
        {
            if(!attached || !pauseAfter) return ValueTask.CompletedTask;
            pauseAfter = false;
        }
        return PauseAsync(context, program, index, descriptor, "after", cancellationToken);
    }

    async ValueTask PauseAsync(VmRunContext context, string program, int index, OpDescriptor descriptor, string phase, CancellationToken cancellationToken)
    {
        lock(syncRoot)
        {
            if(!attached) return;
            resumeRequested = false;
            location = new(program, index, descriptor.Name, phase, context.Identity.RunId, context.Identity.SourceFrameId);
            variables = inspector.Snapshot(context);
            knownVariables = variables;
            knownLocation = location;
            state = "paused";
            revision++;
        }

        try
        {
            while(true)
            {
                await wake.WaitAsync(cancellationToken).ConfigureAwait(false);
                InspectionWork? work;
                lock(syncRoot)
                {
                    if(resumeRequested) return;
                    if(!inspections.TryDequeue(out work)) continue;
                }
                if(work.CancellationToken.IsCancellationRequested)
                {
                    work.Completion.TrySetCanceled(work.CancellationToken);
                    continue;
                }
                try
                {
                    // A local context reference is valid only in this loop. HTTP never receives it or its tensors.
                    DebugInspectionResult result = await inspector.InspectAsync(context, work.Request, cancellationToken).ConfigureAwait(false);
                    work.Completion.TrySetResult(result);
                }
                catch(Exception exception)
                {
                    work.Completion.TrySetException(exception);
                }
            }
        }
        finally
        {
            lock(syncRoot)
            {
                variables = [];
                RejectInspections();
                if(state == "paused") state = "stopped";
            }
        }
    }

    #endregion

    public void Dispose()
    {
        lock(syncRoot)
        {
            disposed = true;
            RejectInspections();
            wake.Dispose();
        }
        Observation.Dispose();
    }

    sealed record InspectionWork(DebugInspectionRequest Request, TaskCompletionSource<DebugInspectionResult> Completion, CancellationToken CancellationToken);
}
