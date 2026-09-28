using System.Diagnostics;
using System.Threading.Channels;
using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: One camera = one <see cref="VmController"/> running the chain <c>[prepare:&lt;camera&gt;, common]</c> with one
/// <c>VmRunContext</c> per frame (ADR-001). The worker owns the frame source (decoder thread), a small drop-oldest
/// frame channel, the controller, its per-source <see cref="VmGlobalMemory"/> (tracker state + zones) and the latest
/// annotated preview. Nothing here is shared with other cameras except the inference endpoints.
///
/// RU: Одна камера = один <see cref="VmController"/>, исполняющий цепочку <c>[prepare:&lt;camera&gt;, common]</c> одним
/// <c>VmRunContext</c> на кадр (ADR-001). Воркер владеет источником кадров (поток декодера), маленьким каналом с
/// вытеснением старых кадров, контроллером, его <see cref="VmGlobalMemory"/> (state трекера + зоны) и последним
/// аннотированным превью. С другими камерами общие только endpoint-ы инференса.
/// </summary>
internal sealed class CameraWorker : IAsyncDisposable
{
    const int FrameChannelCapacity = 2;

    readonly CameraConfig camera;
    readonly MultiCameraConfig config;
    readonly VmController controller;
    readonly IFrameSource source;
    readonly Channel<ISourceFrame> frameChannel;
    readonly object previewSyncRoot = new();
    Mat? latestPreview;

    public CameraWorker(CameraConfig camera, MultiCameraConfig config, SharedInferenceEndpoints endpoints)
    {
        this.camera = camera ?? throw new ArgumentNullException(nameof(camera));
        this.config = config ?? throw new ArgumentNullException(nameof(config));
        ArgumentNullException.ThrowIfNull(endpoints);

        (VmProgram prepare, CameraGeometry geometry) = CameraPrepareProgramFactory.Create(camera, config.Backend);
        VmProgram common = CommonProgramFactory.Create(config, camera, geometry, endpoints);
        Geometry = geometry;

        var globalMemory = new VmGlobalMemory();
        if(camera.Tracking?.StartZone is { } startZone)
            globalMemory.Set(MultiCameraKeys.TrackerStartZone, new TrackRect(startZone.X1, startZone.Y1, startZone.X2, startZone.Y2));

        if(camera.Tracking?.EndZone is { } endZone)
            globalMemory.Set(MultiCameraKeys.TrackerEndZone, new TrackRect(endZone.X1, endZone.Y1, endZone.X2, endZone.Y2));

        var options = new VmControllerOptions(
            SourceId: camera.Id,
            MaxInFlight: camera.MaxInFlight,
            OutputKeys:
            [
                MultiCameraKeys.TrackDetections,
                MultiCameraKeys.Tracks,
                MultiCameraKeys.TextObbRect,
                MultiCameraKeys.TextTrackIds,
                MultiCameraKeys.OcrRoiCount,
                MultiCameraKeys.OcrRecognition,
                MultiCameraKeys.PreviewImage
            ],
            RequireWarmup: true,
            CaptureVariablesInTrace: false);

        controller = new VmController(options, [prepare, common], globalMemory);
        // Surface pool headroom for NVDEC: frames held by in-flight runs plus the frame channel.
        source = FrameSourceFactory.Create(camera.Source, extraHardwareFrames: camera.MaxInFlight + FrameChannelCapacity + 2);

        frameChannel = Channel.CreateBounded<ISourceFrame>(
            new BoundedChannelOptions(FrameChannelCapacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true
            },
            static droppedFrame => droppedFrame.Dispose());
    }

    public string Id => camera.Id;

    public CameraGeometry Geometry { get; }

    public CameraStatistics Statistics { get; } = new();

    public int MaxInFlight => camera.MaxInFlight;

    /// <summary>
    /// Throughput the camera could reach if frames arrived without delay: <c>maxInFlight</c> runs overlap, each takes
    /// the measured run time (GPU work plus queueing in the shared endpoints under the current load).
    /// </summary>
    public double PotentialFps(CameraStatisticsSnapshot snapshot) =>
        snapshot.AverageRunMilliseconds > 0 ? camera.MaxInFlight * 1000.0 / snapshot.AverageRunMilliseconds : 0;

    public string SourceDescription => source.Description;

    public IReadOnlyList<VmProgram> Programs => controller.Programs;

    /// <summary>When set, annotated previews are written there as PNG (first run, then every <see cref="SnapshotEveryRuns"/> runs).</summary>
    public string? SnapshotDirectory { get; set; }

    public int SnapshotEveryRuns { get; set; } = 100;

    /// <summary>Returns a clone of the latest annotated preview or <c>null</c>; the caller owns the clone.</summary>
    public Mat? TakePreviewClone()
    {
        lock(previewSyncRoot)
            return latestPreview?.Clone();
    }

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        Statistics.SetPhase("opening source");
        Task readerTask = Task.Factory.StartNew(
            () => ReadFramesLoop(cancellationToken),
            cancellationToken,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);

        try
        {
            await ProcessFramesLoopAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            frameChannel.Writer.TryComplete();
            try
            {
                await readerTask.ConfigureAwait(false);
            }
            catch(OperationCanceledException)
            {
                // Expected on shutdown.
            }

            DrainChannel();
        }
    }

    #region Frame reading

    void ReadFramesLoop(CancellationToken cancellationToken)
    {
        var pacing = new Stopwatch();

        try
        {
            while(!cancellationToken.IsCancellationRequested)
            {
                pacing.Restart();
                ISourceFrame? frame = source.ReadNext(cancellationToken);
                if(frame is null)
                {
                    Statistics.SetPhase("source exhausted");
                    break;
                }

                Statistics.MarkFrameRead();
                if(!frameChannel.Writer.TryWrite(frame))
                {
                    frame.Dispose();
                    break;
                }

                PaceToSourceFps(pacing.Elapsed, cancellationToken);
            }
        }
        catch(Exception exception) when(exception is not OperationCanceledException)
        {
            Statistics.MarkRunFailed($"source: {exception.Message}");
            Statistics.SetPhase("source failed");
        }
        finally
        {
            frameChannel.Writer.TryComplete();
        }
    }

    void PaceToSourceFps(TimeSpan elapsed, CancellationToken cancellationToken)
    {
        if(!camera.Source.Realtime || source.Fps <= 0)
            return;

        TimeSpan framePeriod = TimeSpan.FromSeconds(1.0 / source.Fps);
        if(elapsed >= framePeriod)
            return;

        // Windows kernel waits have ~15 ms granularity, which turned a 40 ms period into ~60 ms. Coarse-wait most of
        // the remainder, then spin the last two milliseconds on the Stopwatch clock.
        long deadline = Stopwatch.GetTimestamp() + (long)((framePeriod - elapsed).TotalSeconds * Stopwatch.Frequency);
        TimeSpan coarse = framePeriod - elapsed - TimeSpan.FromMilliseconds(2);
        if(coarse > TimeSpan.Zero && cancellationToken.WaitHandle.WaitOne(coarse))
            return;

        var spinner = new SpinWait();
        while(Stopwatch.GetTimestamp() < deadline && !cancellationToken.IsCancellationRequested)
            spinner.SpinOnce(sleep1Threshold: -1);
    }

    #endregion

    #region VM execution

    async Task ProcessFramesLoopAsync(CancellationToken cancellationToken)
    {
        Statistics.SetPhase("warmup");
        ISourceFrame warmupFrame = await frameChannel.Reader.ReadAsync(cancellationToken).ConfigureAwait(false);
        ValidateFrame(warmupFrame);

        // Warmup runs the whole chain, validates device placement after each instruction and builds ONNX kernels /
        // engines for this camera's exact shapes before the hot loop starts.
        VmRunOutcome warmupOutcome = await controller.WarmupAsync(CreateInputs(warmupFrame), cancellationToken).ConfigureAwait(false);
        DisposePreviewOutput(warmupOutcome);
        if(warmupOutcome.Status != VmRunStatus.Completed)
            throw new InvalidOperationException($"Camera '{Id}' warmup failed.", warmupOutcome.Exception);

        Statistics.SetPhase("running");
        var pendingRuns = new List<Task>(camera.MaxInFlight);
        long frameIndex = 0;

        await foreach(ISourceFrame frame in frameChannel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            frameIndex++;

            if(pendingRuns.Count >= camera.MaxInFlight)
            {
                Task finished = await Task.WhenAny(pendingRuns).ConfigureAwait(false);
                pendingRuns.Remove(finished);
            }

            // The frame hands itself to the run context and is disposed with it; a rejected frame is disposed here.
            if(!controller.TryStartRun(CreateInputs(frame), out VmRunHandle handle, sourceFrameId: frameIndex))
            {
                frame.Dispose();
                Statistics.MarkRunRejected();
                continue;
            }

            Statistics.MarkRunStarted();
            pendingRuns.Add(ObserveRunAsync(handle));
        }

        await Task.WhenAll(pendingRuns).ConfigureAwait(false);
    }

    async Task ObserveRunAsync(VmRunHandle handle)
    {
        long startTimestamp = Stopwatch.GetTimestamp();
        VmRunOutcome outcome = await handle.Completion.ConfigureAwait(false);
        TimeSpan duration = Stopwatch.GetElapsedTime(startTimestamp);

        if(outcome.Status != VmRunStatus.Completed)
        {
            DisposePreviewOutput(outcome);
            if(outcome.Status == VmRunStatus.Failed)
                Statistics.MarkRunFailed(outcome.Exception?.GetBaseException().Message ?? outcome.Status.ToString());

            return;
        }

        VmRunOutput output = outcome.Output ?? VmRunOutput.Empty;
        TrackDetection[] detections = output.TryGet(MultiCameraKeys.TrackDetections, out TrackDetection[] detectionValues) ? detectionValues : [];
        TrackedObject[] tracks = output.TryGet(MultiCameraKeys.Tracks, out TrackedObject[] trackValues) ? trackValues : [];
        YoloObb[] textRegions = output.TryGet(MultiCameraKeys.TextObbRect, out YoloObb[] textValues) ? textValues : [];
        int[] textTrackIds = output.TryGet(MultiCameraKeys.TextTrackIds, out int[] textTrackValues) ? textTrackValues : [];
        PaddleOCRRecExtractor.OcrResult[] recognized = ReadRecognition(output);

        Statistics.MarkRunCompleted(duration, detections.Length, tracks, FormatText(recognized, textTrackIds));
        if(outcome.Trace is not null)
            Statistics.AddInstructionTimings(outcome.Trace);

        if(output.TryGet(MultiCameraKeys.PreviewImage, out Mat preview))
        {
            PreviewRenderer.Draw(preview, Geometry, camera, detections, tracks, textRegions, textTrackIds, recognized, Statistics.Snapshot(), PotentialFps(Statistics.Snapshot()));
            SaveSnapshotIfRequested(preview, outcome.Identity.RunId);
            PublishPreview(preview);
        }
    }

    static PaddleOCRRecExtractor.OcrResult[] ReadRecognition(VmRunOutput output)
    {
        if(!output.TryGet(MultiCameraKeys.OcrRecognition, out PaddleOCRRecExtractor.OcrResult[] results))
            return [];

        // With fixed ROI capacity the recognition batch carries padding slots; only the first "actual count" are real.
        if(output.TryGet(MultiCameraKeys.OcrRoiCount, out int actualCount) && actualCount < results.Length)
            return results[..Math.Max(0, actualCount)];

        return results;
    }

    /// <summary>Text grouped by track: "T3: ORENBURG | OM-450KG; T4: ...".</summary>
    static string? FormatText(PaddleOCRRecExtractor.OcrResult[] recognized, int[] textTrackIds)
    {
        if(recognized.Length == 0)
            return null;

        var byTrack = new SortedDictionary<int, List<string>>();
        for(int index = 0; index < recognized.Length; index++)
        {
            string text = recognized[index].Standard;
            if(string.IsNullOrWhiteSpace(text))
                continue;

            int trackId = index < textTrackIds.Length ? textTrackIds[index] : -1;
            if(!byTrack.TryGetValue(trackId, out List<string>? texts))
                byTrack[trackId] = texts = [];

            texts.Add(text);
        }

        return byTrack.Count == 0
            ? null
            : string.Join("; ", byTrack.Select(static pair => $"T{pair.Key}: {string.Join(" | ", pair.Value)}"));
    }

    static VmRunInputs CreateInputs(ISourceFrame frame)
    {
        var inputs = new VmRunInputs();
        frame.AddTo(inputs);
        return inputs;
    }


    void ValidateFrame(ISourceFrame frame)
    {
        if(frame.Width != camera.Resolution.Width || frame.Height != camera.Resolution.Height)
        {
            throw new InvalidOperationException(
                $"Camera '{Id}' delivers {frame.Width}x{frame.Height} frames but the config declares " +
                $"{camera.Resolution.Width}x{camera.Resolution.Height}; the geometry chain is built for the declared size.");
        }
    }

    #endregion

    #region Preview

    void PublishPreview(Mat preview)
    {
        Mat? previous;
        lock(previewSyncRoot)
        {
            previous = latestPreview;
            latestPreview = preview;
        }

        previous?.Dispose();
    }

    void SaveSnapshotIfRequested(Mat preview, long runId)
    {
        if(SnapshotDirectory is null || (runId != 1 && runId % SnapshotEveryRuns != 0))
            return;

        Directory.CreateDirectory(SnapshotDirectory);
        Cv2.ImWrite(Path.Combine(SnapshotDirectory, $"{Id}_run{runId:D6}.png"), preview);
    }

    static void DisposePreviewOutput(VmRunOutcome outcome)
    {
        if(outcome.Output is not null && outcome.Output.TryGet(MultiCameraKeys.PreviewImage, out Mat preview))
            preview.Dispose();
    }

    #endregion

    void DrainChannel()
    {
        while(frameChannel.Reader.TryRead(out ISourceFrame? frame))
            frame.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        await controller.DisposeAsync().ConfigureAwait(false);
        source.Dispose();
        DrainChannel();

        lock(previewSyncRoot)
        {
            latestPreview?.Dispose();
            latestPreview = null;
        }
    }
}
