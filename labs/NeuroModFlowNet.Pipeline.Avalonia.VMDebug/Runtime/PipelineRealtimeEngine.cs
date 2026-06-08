using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime.Debug;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Runs source capture and submits accepted work items into the VM debug program.
/// </summary>
/// <remarks>
/// This class is the host loop, not the VM itself. It owns VideoCapture, playback, and frame-drop policy, then passes
/// named inputs into VmController. The VM does not know whether source.frame came from a camera, a file, RTSP, or
/// an outer event-system module.
/// </remarks>
internal sealed class PipelineRealtimeEngine : IDisposable
{
    const string PrimarySourceId = "camera:0";
    const string PrimarySourceConfigKey = VideoCaptureConfig.DefaultConfigKey;
    readonly RealTimeAvaloniaSettings settings;
    readonly RecognitionOptions recognitionOptions;
    readonly AvaloniaJsonConfig jsonConfig;
    readonly VmDebugOperationRegistry operationRegistry;
    readonly ManualPipelineInstructionDebugGate instructionDebugGate = new();
    readonly object lifecycleSyncRoot = new();
    readonly object playbackSyncRoot = new();
    readonly object programSyncRoot = new();

    PipelineModelResources? resources;
    VmController? controller;
    CancellationTokenSource? cancellationTokenSource;
    Task? workerTask;
    IReadOnlyList<string> activeOperations = [];
    int programVersion;
    bool isPaused;
    int pendingStepCount;
    bool disposed;
    long droppedRuns;
    int initializedBatchSize;
    int initializedRecognitionShapeVersion;

    public PipelineRealtimeEngine(
        RealTimeAvaloniaSettings settings,
        RecognitionOptions recognitionOptions,
        AvaloniaJsonConfig jsonConfig)
    {
        this.settings = settings;
        this.recognitionOptions = recognitionOptions;
        this.jsonConfig = jsonConfig;
        operationRegistry = new VmDebugOperationRegistry(settings, recognitionOptions, jsonConfig);
        instructionDebugGate.Stopped += OnInstructionDebugGateStopped;
        initializedBatchSize = recognitionOptions.BatchSize;
        initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
    }

    public event EventHandler<PipelineFrameResultData>? FrameReady;

    public event EventHandler<string>? StatusChanged;

    public event EventHandler<OpDebugPoint>? InstructionStopped;

    public bool IsRunning => workerTask is { IsCompleted: false };

    public IReadOnlySet<string> KnownOperationNames => operationRegistry.OperationNames;

    public void ConfigureProgram(IReadOnlyList<VmDebugCompiledStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        lock(programSyncRoot)
        {
            activeOperations = steps
                .Where(step => step.IsKnown)
                .Select(step => step.Operation)
                .ToArray();
            programVersion++;
        }
    }

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;
        instructionDebugGate.Stopped -= OnInstructionDebugGateStopped;
        StopAsync().GetAwaiter().GetResult();
    }

    public Task StartAsync()
    {
        ThrowIfDisposed();

        lock(lifecycleSyncRoot)
        {
            if(IsRunning)
                return Task.CompletedTask;

            StatusChanged?.Invoke(this, "Starting...");
            ResetPlaybackState();
            cancellationTokenSource = new CancellationTokenSource();
            workerTask = Task.Run(() => RunAsync(cancellationTokenSource.Token));
        }

        return Task.CompletedTask;
    }

    public async Task StopAsync()
    {
        CancellationTokenSource? tokenSource;
        Task? task;

        lock(lifecycleSyncRoot)
        {
            tokenSource = cancellationTokenSource;
            task = workerTask;
            cancellationTokenSource = null;
            workerTask = null;
        }

        if(tokenSource is not null)
        {
            await tokenSource.CancelAsync().ConfigureAwait(false);

            if(task is not null)
            {
                try
                {
                    await task.ConfigureAwait(false);
                }
                catch(OperationCanceledException)
                {
                }
            }

            tokenSource.Dispose();
        }

        if(controller is not null)
            await controller.DisposeAsync().ConfigureAwait(false);
        controller = null;

        resources?.Dispose();
        resources = null;
        ResetPlaybackState();
        StatusChanged?.Invoke(this, "Stopped");
    }

    public void Pause()
    {
        ThrowIfDisposed();

        lock(playbackSyncRoot)
        {
            isPaused = true;
            pendingStepCount = 0;
        }

        StatusChanged?.Invoke(this, "Paused");
    }

    public void Play()
    {
        ThrowIfDisposed();

        lock(playbackSyncRoot)
        {
            isPaused = false;
            pendingStepCount = 0;
        }

        instructionDebugGate.Disable();
        StatusChanged?.Invoke(this, IsRunning ? "Running" : "Ready");
    }

    public void StepFrame()
    {
        ThrowIfDisposed();

        lock(playbackSyncRoot)
        {
            isPaused = true;
            pendingStepCount++;
        }

        StatusChanged?.Invoke(this, "Paused: next frame requested");
    }

    public void StepFrameWithInstructionDebug()
    {
        ThrowIfDisposed();

        lock(playbackSyncRoot)
        {
            isPaused = true;
            pendingStepCount++;
        }

        instructionDebugGate.EnableForNextRun();
        StatusChanged?.Invoke(this, "Paused: debug frame requested");
    }

    public bool TryResumeOneInstruction()
    {
        ThrowIfDisposed();
        bool resumed = instructionDebugGate.TryResumeOneInstruction();
        if(resumed)
            StatusChanged?.Invoke(this, "VM step resumed");

        return resumed;
    }

    async Task RunAsync(CancellationToken cancellationToken)
    {
        PipelineModelResources? modelResources = await TryCreateResourcesAsync(cancellationToken).ConfigureAwait(false);
        resources = modelResources;
        (IReadOnlyList<string> operations, int controllerProgramVersion) = SnapshotProgram();
        VmProgram program = operationRegistry.BuildProgram(operations, modelResources);
        controller = CreateController(PrimarySourceId, program);

        using VideoCapture primaryCapture = VideoCaptureConfig.CreateFromConfig(PrimarySourceConfigKey);
        using var primarySourceFrame = new Mat();

        long primarySourceFrameId = 0;
        StatusChanged?.Invoke(this, modelResources is null ? "Camera only" : "Running");

        while(!cancellationToken.IsCancellationRequested)
        {
            await EnsureRecognitionShapeAsync(cancellationToken).ConfigureAwait(false);
            (controller, controllerProgramVersion) = await RebuildControllerIfProgramChangedAsync(
                controller,
                controllerProgramVersion,
                modelResources,
                cancellationToken).ConfigureAwait(false);

            if(!TryConsumeFrameRequest())
            {
                await Task.Delay(15, cancellationToken).ConfigureAwait(false);
                continue;
            }

            using Mat primaryFrame = CaptureFrame(primaryCapture, primarySourceFrame);
            SubmitFrame(controller, PrimarySourceId, primaryFrame, primarySourceFrameId++);

            await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        }
    }

    async Task<PipelineModelResources?> TryCreateResourcesAsync(CancellationToken cancellationToken)
    {
        try
        {
            StatusChanged?.Invoke(this, $"Initializing models: YOLO={settings.InferenceBackend}, PaddleDet={settings.PaddleDetInferenceBackend}, PaddleRec={settings.PaddleRecInferenceBackend}...");
            PipelineModelResources modelResources = await PipelineModelResources
                .CreateAsync(settings, recognitionOptions, jsonConfig, new DemoOnnxAssetResolver())
                .ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            initializedBatchSize = recognitionOptions.BatchSize;
            initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
            return modelResources;
        }
        catch(Exception exception) when(exception is not OperationCanceledException)
        {
            StatusChanged?.Invoke(this, $"Camera only. Inference initialization failed: {exception.Message}");
            return null;
        }
    }

    VmController CreateController(string sourceId, VmProgram program) =>
        new(
            new VmControllerOptions(
                sourceId,
                MaxInFlight: 2,
                OutputKeys: [PipelineAvaloniaKeys.FrameResult],
                DebugGate: instructionDebugGate),
            program);

    async ValueTask<(VmController Controller, int ProgramVersion)> RebuildControllerIfProgramChangedAsync(
        VmController currentController,
        int currentProgramVersion,
        PipelineModelResources? modelResources,
        CancellationToken cancellationToken)
    {
        (IReadOnlyList<string> operations, int latestProgramVersion) = SnapshotProgram();
        if(latestProgramVersion == currentProgramVersion)
            return (currentController, currentProgramVersion);

        StatusChanged?.Invoke(this, "Rebuilding VM program...");
        await currentController.DisposeAsync().ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        VmProgram program = operationRegistry.BuildProgram(operations, modelResources);
        VmController nextController = CreateController(PrimarySourceId, program);
        controller = nextController;
        StatusChanged?.Invoke(this, modelResources is null ? "Camera only" : "Running");
        return (nextController, latestProgramVersion);
    }

    (IReadOnlyList<string> Operations, int Version) SnapshotProgram()
    {
        lock(programSyncRoot)
        {
            if(activeOperations.Count == 0)
                activeOperations = operationRegistry.DefaultOperationNames;

            return (activeOperations, programVersion);
        }
    }

    async ValueTask EnsureRecognitionShapeAsync(CancellationToken cancellationToken)
    {
        if(resources is null)
            return;

        if(initializedBatchSize == recognitionOptions.BatchSize &&
           initializedRecognitionShapeVersion == recognitionOptions.RecognitionShapeVersion)
            return;

        await resources.EnsureRecognitionBatchAsync(recognitionOptions, cancellationToken).ConfigureAwait(false);
        initializedBatchSize = recognitionOptions.BatchSize;
        initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
    }

    async Task ObserveRunCompletionAsync(VmRunHandle handle)
    {
        VmRunOutcome outcome = await handle.Completion.ConfigureAwait(false);
        if(outcome.Status is VmRunStatus.Canceled)
            instructionDebugGate.Disable();

        if(outcome.Status == VmRunStatus.Failed && outcome.Exception is not null)
        {
            instructionDebugGate.Disable();
            StatusChanged?.Invoke(this, $"Run {outcome.Identity.RunId} failed: {outcome.Exception.Message}");
            return;
        }

        if(outcome.Status == VmRunStatus.Completed &&
           outcome.Output?.TryGet(PipelineAvaloniaKeys.FrameResult, out PipelineFrameResultData? frameData) == true &&
           frameData is not null)
        {
            instructionDebugGate.Disable();
            frameData.AttachTrace(outcome.Trace);
            PublishFrame(frameData);
        }
    }

    void SubmitFrame(
        VmController activeController,
        string sourceId,
        Mat capturedFrame,
        long sourceFrameId)
    {
        Mat ownedFrame = capturedFrame.Clone();
        var timing = new PipelineFrameTiming();

        var inputs = new VmRunInputs()
            .Add(PipelineAvaloniaKeys.SourceId, sourceId)
            .Add(PipelineAvaloniaKeys.SourceFrame, ownedFrame, disposeWithContext: true)
            .Add(PipelineAvaloniaKeys.FrameTiming, timing);

        bool started = activeController.TryStartRun(
            inputs,
            out VmRunHandle handle,
            sourceFrameId);

        if(started)
            _ = ObserveRunCompletionAsync(handle);
        else
        {
            ownedFrame.Dispose();
            Interlocked.Increment(ref droppedRuns);
        }
    }

    void PublishFrame(PipelineFrameResultData frameData)
    {
        EventHandler<PipelineFrameResultData>? handler = FrameReady;
        if(handler is null)
        {
            frameData.Dispose();
            return;
        }

        handler.Invoke(this, frameData);
    }

    void OnInstructionDebugGateStopped(object? sender, OpDebugPoint point)
    {
        StatusChanged?.Invoke(this, $"VM stopped before {point.InstructionIndex:00} {point.Operation}");
        InstructionStopped?.Invoke(this, point);
    }

    static Mat CaptureFrame(VideoCapture capture, Mat reusableFrame)
    {
        capture.Read(reusableFrame);
        if(!reusableFrame.Empty())
            return reusableFrame.Clone();

        // File sources can still be used through VideoSource. Rewind once so the single VM input keeps moving in debug.
        capture.Set(VideoCaptureProperties.PosFrames, 0);
        capture.Read(reusableFrame);
        if(!reusableFrame.Empty())
            return reusableFrame.Clone();

        var dummy = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(dummy, "NO SIGNAL", new Point(480, 360), HersheyFonts.HersheySimplex, 1.5, new Scalar(70, 70, 70), 2);
        return dummy;
    }

    bool TryConsumeFrameRequest()
    {
        lock(playbackSyncRoot)
        {
            if(!isPaused)
                return true;

            if(pendingStepCount <= 0)
                return false;

            pendingStepCount--;
            return true;
        }
    }

    void ResetPlaybackState()
    {
        lock(playbackSyncRoot)
        {
            isPaused = false;
            pendingStepCount = 0;
        }
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(PipelineRealtimeEngine));
    }
}
