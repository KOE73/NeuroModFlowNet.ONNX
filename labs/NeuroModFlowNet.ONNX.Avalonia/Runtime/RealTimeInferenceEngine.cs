using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;
using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Runs camera capture, neural network inference, OCR ROI extraction, recognition, and frame packaging in the background.
/// RU: Выполняет в фоне захват камеры, inference нейросетей, вырезание OCR ROI, recognition и упаковку данных кадра.
/// </summary>
/// <remarks>
/// EN: The engine does not draw UI directly. It raises <see cref="FrameReady"/> with one frame data object and raises
/// <see cref="StatusChanged"/> for user-visible state messages. Consumers must marshal events to the UI thread before
/// touching Avalonia controls.
/// RU: Engine не рисует UI напрямую. Он отдает <see cref="FrameReady"/> с данными одного кадра и <see cref="StatusChanged"/>
/// с сообщениями состояния. Потребители должны переключаться на UI thread перед обращением к Avalonia controls.
/// </remarks>
public sealed class RealTimeInferenceEngine : IDisposable
{
    #region Const and Fields

    const int RecognitionLabelOffsetY = 6;

    readonly RealTimeAvaloniaSettings settings;
    readonly RecognitionOptions recognitionOptions;
    readonly AvaloniaJsonConfig jsonConfig;
    readonly object lifecycleSyncRoot = new();
    readonly object playbackSyncRoot = new();

    CancellationTokenSource? cancellationTokenSource;
    Task? workerTask;
    bool isPaused;
    int pendingStepCount;
    bool disposed;

    #endregion

    #region Lifecycle

    public RealTimeInferenceEngine(RealTimeAvaloniaSettings settings, RecognitionOptions recognitionOptions, AvaloniaJsonConfig jsonConfig)
    {
        this.settings = settings;
        this.recognitionOptions = recognitionOptions;
        this.jsonConfig = jsonConfig;
    }

    public event EventHandler<RealTimeOneFrameData>? FrameReady;
    public event EventHandler<string>? StatusChanged;
    public event EventHandler<IReadOnlyList<RuntimeModelInfo>>? ModelInfoChanged;

    public bool IsRunning => workerTask is { IsCompleted: false };
    public bool IsPaused
    {
        get
        {
            lock(playbackSyncRoot)
                return isPaused;
        }
    }

    public void Dispose()
    {
        if(disposed) return;

        disposed = true;
        StopAsync().GetAwaiter().GetResult();
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(RealTimeInferenceEngine));
    }

    #endregion

    #region Public Control

    public Task StartAsync()
    {
        ThrowIfDisposed();

        lock(lifecycleSyncRoot)
        {
            if(IsRunning) return Task.CompletedTask;

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

        if(tokenSource is null) return;

        await tokenSource.CancelAsync();

        if(task is not null)
        {
            try
            {
                await task;
            }
            catch(OperationCanceledException)
            {
            }
        }

        tokenSource.Dispose();
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

    void ResetPlaybackState()
    {
        lock(playbackSyncRoot)
        {
            isPaused = false;
            pendingStepCount = 0;
        }
    }

    #endregion

    #region Capture Loop

    async Task RunAsync(CancellationToken cancellationToken)
    {
        using VideoCapture capture = VideoCaptureConfig.CreateFromConfig();
        using var sourceFrame = new Mat();
        using var heldFrame = new Mat();
        using InferenceResources? resources = await TryCreateInferenceResourcesAsync(cancellationToken);

        int initializedBatchSize = recognitionOptions.BatchSize;
        int initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
        long lastFrameTicks = System.Diagnostics.Stopwatch.GetTimestamp();

        while(!cancellationToken.IsCancellationRequested)
        {
            using Mat capturedFrame = CaptureOrReplayFrame(capture, sourceFrame, heldFrame);
            using Mat resizedFrame = ResizeFrame(capturedFrame, recognitionOptions.FrameWidth);

            RealTimeOneFrameData update = resources is null
                ? BuildCameraOnlyUpdate(capturedFrame, ref lastFrameTicks)
                : ProcessInferenceFrame(resources, capturedFrame, resizedFrame, ref initializedBatchSize, ref initializedRecognitionShapeVersion, ref lastFrameTicks);

            FrameReady?.Invoke(this, update);
            await Task.Delay(1, cancellationToken);
        }
    }

    #endregion

    #region Inference Initialization

    async Task<InferenceResources?> TryCreateInferenceResourcesAsync(CancellationToken cancellationToken)
    {
        try
        {
            StatusChanged?.Invoke(this, "Initializing models...");
            InferenceResources resources = await InferenceResources.CreateAsync(settings, recognitionOptions, jsonConfig);
            ModelInfoChanged?.Invoke(this, resources.ModelInfos);
            cancellationToken.ThrowIfCancellationRequested();
            StatusChanged?.Invoke(this, IsPaused ? "Paused" : "Running");
            return resources;
        }
        catch(Exception exception) when(exception is not OperationCanceledException)
        {
            StatusChanged?.Invoke(this, $"Camera only. Inference initialization failed: {exception.Message}");
            return null;
        }
    }

    #endregion

    #region Frame Inference

    RealTimeOneFrameData ProcessInferenceFrame(
        InferenceResources resources,
        Mat sourceFrame,
        Mat resizedFrame,
        ref int initializedBatchSize,
        ref int initializedRecognitionShapeVersion,
        ref long lastFrameTicks)
    {
        if(initializedBatchSize != recognitionOptions.BatchSize ||
           initializedRecognitionShapeVersion != recognitionOptions.RecognitionShapeVersion)
        {
            resources.EnsureRecognitionBatch(recognitionOptions);
            ModelInfoChanged?.Invoke(this, resources.ModelInfos);
            initializedBatchSize = recognitionOptions.BatchSize;
            initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
        }

        bool ocrEnabled = recognitionOptions.InferenceSelection.OcrEnabled;
        bool boxDetectionEnabled = recognitionOptions.InferenceSelection.BoxDetectionEnabled;
        bool obbDetectionEnabled = recognitionOptions.InferenceSelection.ObbDetectionEnabled;
        bool segmentationEnabled = recognitionOptions.InferenceSelection.SegmentationEnabled;
        bool classificationEnabled = recognitionOptions.InferenceSelection.ClassificationEnabled;
        bool poseEnabled = recognitionOptions.InferenceSelection.PoseEnabled;
        bool obbPredictionNeeded = ocrEnabled || obbDetectionEnabled;

        if(!ocrEnabled &&
           !boxDetectionEnabled &&
           !obbDetectionEnabled &&
           !segmentationEnabled &&
           !classificationEnabled &&
           !poseEnabled)
            return BuildCameraOnlyUpdate(sourceFrame, ref lastFrameTicks);

        var metricsStopwatch = System.Diagnostics.Stopwatch.StartNew();
        using Mat letterboxed = resizedFrame.Letterbox(settings.InputSize, settings.InputSize, out LetterboxInfo letterboxInfo);
        LetterboxInfo sourceLetterboxInfo = CreateSourceLetterboxInfo(letterboxInfo, resizedFrame, sourceFrame);
        if(ocrEnabled)
            resources.EnsureDetFrameShape(letterboxed);

        long detectionStartTicks = metricsStopwatch.ElapsedTicks;
        IDetectionResult<YoloBox>? boxResult = boxDetectionEnabled ? resources.RunnerBox.Predict(letterboxed) : null;
        IDetectionResult<YoloObb>? obbResult = obbPredictionNeeded ? resources.RunnerObb.Predict(letterboxed) : null;
        Mat? detScoreMap = ocrEnabled ? resources.RunnerDet.Predict(letterboxed) : null;
        IBatchedResult? segmentationResult = segmentationEnabled ? resources.RunnerSeg.Predict(letterboxed) : null;
        IBatchedResult? classificationResult = classificationEnabled ? resources.RunnerCls.Predict(letterboxed) : null;
        IDetectionResult<YoloPose>? poseResult = poseEnabled ? resources.RunnerPose.Predict(letterboxed) : null;
        double detectionMilliseconds = ElapsedMilliseconds(metricsStopwatch, detectionStartTicks);

        try
        {
            YoloBox[] boxBoxes = boxResult?.GetBatch(0).ToArray() ?? [];
            YoloObb[] obbBoxes = obbResult?.GetBatch(0).ToArray() ?? [];
            List<OcrQuadRegion> detRegions = detScoreMap is null
                ? []
                : PrepareDetSourceRegions(detScoreMap, sourceLetterboxInfo);

            long roiStartTicks = metricsStopwatch.ElapsedTicks;
            List<(Mat Roi, Point LabelPoint, RoiHeightDebugData RoiHeightDebug)> preparedImages = ocrEnabled
                ? PrepareRecognitionImages(sourceFrame, obbBoxes, sourceLetterboxInfo)
                : [];
            double roiMilliseconds = ElapsedMilliseconds(metricsStopwatch, roiStartTicks);

            try
            {
                long recognitionStartTicks = metricsStopwatch.ElapsedTicks;
                List<(string Text, Point LabelPoint, Mat Roi, RoiHeightDebugData RoiHeightDebug)> recognitionRows = ocrEnabled
                    ? PredictRecognition(resources, preparedImages)
                    : [];
                double recognitionMilliseconds = ElapsedMilliseconds(metricsStopwatch, recognitionStartTicks);

                using Mat visualizedFrame = sourceFrame.Clone();
                DrawAdditionalInferenceResults(
                    visualizedFrame,
                    segmentationResult,
                    classificationResult,
                    poseResult,
                    sourceLetterboxInfo,
                    resources,
                    segmentationEnabled,
                    classificationEnabled,
                    poseEnabled);
                Mat displayFrame = ToBgra(visualizedFrame);
                Mat detDisplayFrame = detScoreMap is null
                    ? CreateEmptyDetPreview(sourceFrame)
                    : CreateDetPreview(detScoreMap);

                FrameOverlaySnapshot overlay = BuildOverlay(
                    sourceFrame,
                    boxBoxes,
                    obbBoxes,
                    detRegions,
                    sourceLetterboxInfo,
                    resources,
                    recognitionRows,
                    boxDetectionEnabled,
                    ocrEnabled || obbDetectionEnabled);

                return new RealTimeOneFrameData(
                    displayFrame,
                    detDisplayFrame,
                    overlay,
                    recognitionRows
                        .Take(12)
                        .Select(row => new RealTimeRecognitionItemData(row.Roi.Clone(), row.Text, recognitionOptions.RoiDisplayScale, row.RoiHeightDebug))
                        .ToArray(),
                    new RealTimeMetricsSnapshot(
                        CalculateFps(ref lastFrameTicks),
                        detectionMilliseconds,
                        roiMilliseconds,
                        recognitionMilliseconds,
                        preparedImages.Count));
            }
            finally
            {
                foreach((Mat roi, _, _) in preparedImages)
                    roi.Dispose();
            }
        }
        finally
        {
            (boxResult as IDisposable)?.Dispose();
            (obbResult as IDisposable)?.Dispose();
            (segmentationResult as IDisposable)?.Dispose();
            (classificationResult as IDisposable)?.Dispose();
            (poseResult as IDisposable)?.Dispose();
            detScoreMap?.Dispose();
        }
    }

    #endregion

    #region Recognition ROI Preparation

    List<(Mat Roi, Point LabelPoint, RoiHeightDebugData RoiHeightDebug)> PrepareRecognitionImages(
        Mat sourceFrame,
        ReadOnlySpan<YoloObb> boxes,
        LetterboxInfo letterboxInfo)
    {
        var mapper = LetterboxCoordinateMapper.Create(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);

        // Most realtime OCR frames have a small number of text boxes. Keep the detector-to-region
        // bridge stack-only; the common postprocessor caches geometry for all enabled cleanup features.
        Span<OcrQuadRegion> mappedSourceRegions = boxes.Length <= 1000
            ? stackalloc OcrQuadRegion[boxes.Length]
            : new OcrQuadRegion[boxes.Length];
        Span<RoiHeightDebugData> mappedHeightDebug = boxes.Length <= 1000
            ? stackalloc RoiHeightDebugData[boxes.Length]
            : new RoiHeightDebugData[boxes.Length];

        for(int index = 0; index < boxes.Length; index++)
        {
            // Adaptive ROI padding must be tuned in the same coordinate system that the cropper uses.
            // Map the unscaled detector box to the original source frame first, derive its real text-line
            // height there, then map the scaled box through the same source-frame mapper.
            OcrQuadRegion unscaledSourceRegion = YoloObbOcrRegionMapper.MapToSourceRegion(boxes[index], mapper);
            float sourceRegionHeight = GetRegionHeight(unscaledSourceRegion);
            RoiHeightDebugData heightDebug = recognitionOptions.CalculateRoiHeightDebug(sourceRegionHeight);
            mappedSourceRegions[index] = YoloObbOcrRegionMapper.MapToSourceRegion(boxes[index], mapper, heightDebug.Scale);
            mappedHeightDebug[index] = heightDebug;
        }

        List<OcrQuadRegion> sourceRegions = [];
        OcrRegionPostprocessor.Shared.Process(
            mappedSourceRegions,
            recognitionOptions.CreateRegionPostprocessorOptions(),
            sourceRegions);

        var options = new TextRegionExtractionOptions(
            recognitionOptions.RecognitionInputWidth,
            recognitionOptions.RecognitionInputHeight,
            recognitionOptions.ProcessingStage);

        List<(Mat Roi, Point LabelPoint, RoiHeightDebugData RoiHeightDebug)> preparedImages = [];

        foreach(OcrQuadRegion sourceRegion in sourceRegions)
        {
            if(!NaiveTextRegionExtractor.Shared.TryExtract(
                sourceFrame,
                sourceRegion,
                options,
                out Mat? recognitionRoi)) continue;
            if(recognitionRoi is null) continue;

            RoiHeightDebugData heightDebug = FindNearestHeightDebug(sourceRegion, mappedSourceRegions, mappedHeightDebug);
            preparedImages.Add((recognitionRoi, GetRecognitionLabelPoint(sourceRegion, sourceFrame), heightDebug));
        }

        return preparedImages;
    }

    static RoiHeightDebugData FindNearestHeightDebug(
        OcrQuadRegion displayRegion,
        ReadOnlySpan<OcrQuadRegion> sourceRegions,
        ReadOnlySpan<RoiHeightDebugData> heightDebugData)
    {
        if(sourceRegions.Length == 0 || heightDebugData.Length == 0)
            return RoiHeightDebugData.Empty;

        Point2f displayCenter = GetRegionCenter(displayRegion);
        float bestDistanceSquared = float.PositiveInfinity;
        int bestIndex = 0;

        for(int index = 0; index < sourceRegions.Length; index++)
        {
            Point2f sourceCenter = GetRegionCenter(sourceRegions[index]);
            float dx = displayCenter.X - sourceCenter.X;
            float dy = displayCenter.Y - sourceCenter.Y;
            float distanceSquared = dx * dx + dy * dy;
            if(distanceSquared >= bestDistanceSquared) continue;

            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }

        // After suppression or line merge the resulting quad may not have a strict one-to-one source OBB.
        // Nearest-center debug keeps the UI cheap and still shows which original height formula drove the ROI.
        return heightDebugData[Math.Min(bestIndex, heightDebugData.Length - 1)];
    }

    static Point2f GetRegionCenter(OcrQuadRegion region) =>
        new(
            (region.X0 + region.X1 + region.X2 + region.X3) * 0.25f,
            (region.Y0 + region.Y1 + region.Y2 + region.Y3) * 0.25f);

    static float GetRegionHeight(OcrQuadRegion region) =>
        MathF.Max(
            Distance(region.Point0, region.Point3),
            Distance(region.Point1, region.Point2));

    static float Distance(Point2f left, Point2f right)
    {
        float dx = left.X - right.X;
        float dy = left.Y - right.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    static LetterboxInfo CreateSourceLetterboxInfo(LetterboxInfo displayLetterboxInfo, Mat displayFrame, Mat sourceFrame)
    {
        if(displayFrame.Width == sourceFrame.Width && displayFrame.Height == sourceFrame.Height)
            return displayLetterboxInfo;

        float displayScale = displayFrame.Width / (float)sourceFrame.Width;

        return new LetterboxInfo
        {
            Ratio = displayLetterboxInfo.Ratio * displayScale,
            OffsetX = displayLetterboxInfo.OffsetX,
            OffsetY = displayLetterboxInfo.OffsetY,
            SourceWidth = sourceFrame.Width,
            SourceHeight = sourceFrame.Height,
            TargetWidth = displayLetterboxInfo.TargetWidth,
            TargetHeight = displayLetterboxInfo.TargetHeight,
        };
    }

    static Point GetRecognitionLabelPoint(OcrQuadRegion sourceRegion, Mat sourceMat)
    {
        Rect bounds = ClipRect(GetBoundingRect(sourceRegion), sourceMat.Width, sourceMat.Height);
        return new Point(bounds.X, Math.Max(0, bounds.Y - RecognitionLabelOffsetY));
    }

    static Rect GetBoundingRect(OcrQuadRegion sourceRegion)
    {
        float left = Math.Min(Math.Min(sourceRegion.X0, sourceRegion.X1), Math.Min(sourceRegion.X2, sourceRegion.X3));
        float top = Math.Min(Math.Min(sourceRegion.Y0, sourceRegion.Y1), Math.Min(sourceRegion.Y2, sourceRegion.Y3));
        float right = Math.Max(Math.Max(sourceRegion.X0, sourceRegion.X1), Math.Max(sourceRegion.X2, sourceRegion.X3));
        float bottom = Math.Max(Math.Max(sourceRegion.Y0, sourceRegion.Y1), Math.Max(sourceRegion.Y2, sourceRegion.Y3));

        int x = (int)Math.Floor(left);
        int y = (int)Math.Floor(top);
        return new Rect(
            x,
            y,
            Math.Max(1, (int)Math.Ceiling(right) - x),
            Math.Max(1, (int)Math.Ceiling(bottom) - y));
    }

    static Rect ClipRect(Rect rect, int width, int height)
    {
        int x = Math.Clamp(rect.X, 0, width - 1);
        int y = Math.Clamp(rect.Y, 0, height - 1);
        int right = Math.Clamp(rect.Right, x + 1, width);
        int bottom = Math.Clamp(rect.Bottom, y + 1, height);
        return new Rect(x, y, right - x, bottom - y);
    }

    #endregion

    #region Recognition Prediction

    List<(string Text, Point LabelPoint, Mat Roi, RoiHeightDebugData RoiHeightDebug)> PredictRecognition(
        InferenceResources resources,
        List<(Mat Roi, Point LabelPoint, RoiHeightDebugData RoiHeightDebug)> preparedImages)
    {
        List<(string Text, Point LabelPoint, Mat Roi, RoiHeightDebugData RoiHeightDebug)> outputResults = [];

        for(int offset = 0; offset < preparedImages.Count; offset += recognitionOptions.BatchSize)
        {
            int itemCount = Math.Min(recognitionOptions.BatchSize, preparedImages.Count - offset);
            List<Mat> batchImages = new(itemCount);

            for(int index = 0; index < itemCount; index++)
                batchImages.Add(preparedImages[offset + index].Roi);

            List<PaddleOCRRecExtractor.OcrResult> batchResults = resources.RunnerRec!.Predict(batchImages);
            int resultCount = Math.Min(itemCount, batchResults.Count);

            for(int index = 0; index < resultCount; index++)
            {
                PaddleOCRRecExtractor.OcrResult recognitionResult = batchResults[index];
                if(recognitionResult.IsEmpty) continue;

                string recognitionText = recognitionOptions.FormatRecognitionText(recognitionResult);
                (Mat roi, Point labelPoint, RoiHeightDebugData heightDebug) = preparedImages[offset + index];
                outputResults.Add((recognitionText, labelPoint, roi, heightDebug));
            }
        }

        return outputResults;
    }

    #endregion

    #region PaddleOCR Det Region Preparation

    List<OcrQuadRegion> PrepareDetSourceRegions(Mat detScoreMap, LetterboxInfo sourceLetterboxInfo)
    {
        List<OcrQuadRegion> modelRegions = [];
        PaddleOCRDetMaskRegionExtractor.Shared.Extract(
            detScoreMap,
            jsonConfig.Postprocessing.DetMask,
            modelRegions);

        if(modelRegions.Count == 0)
            return [];

        var mapper = LetterboxCoordinateMapper.Create(
            sourceLetterboxInfo.Ratio,
            sourceLetterboxInfo.OffsetX,
            sourceLetterboxInfo.OffsetY);

        Span<Point2f> modelPoints = stackalloc Point2f[4];
        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        List<OcrQuadRegion> mappedRegions = new(modelRegions.Count);

        foreach(OcrQuadRegion modelRegion in modelRegions)
        {
            modelRegion.CopyTo(modelPoints);
            mapper.MapPointsToSource(modelPoints, sourcePoints);
            mappedRegions.Add(OcrQuadRegion.FromPoints(sourcePoints));
        }

        List<OcrQuadRegion> sourceRegions = [];
        OcrRegionPostprocessor.Shared.Process(
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(mappedRegions),
            recognitionOptions.CreateRegionPostprocessorOptions(),
            sourceRegions);

        return sourceRegions;
    }

    #endregion

    #region Overlay Preparation

    FrameOverlaySnapshot BuildOverlay(
        Mat frame,
        ReadOnlySpan<YoloBox> boxDetections,
        ReadOnlySpan<YoloObb> obbDetections,
        IReadOnlyList<OcrQuadRegion> detRegions,
        LetterboxInfo letterboxInfo,
        InferenceResources resources,
        IReadOnlyList<(string Text, Point LabelPoint, Mat Roi, RoiHeightDebugData RoiHeightDebug)> recognitionRows,
        bool drawBoxDetection,
        bool drawObbDetection)
    {
        var mapper = LetterboxCoordinateMapper.Create(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);
        List<OverlayObb> overlayBoxes = new(boxDetections.Length + obbDetections.Length + detRegions.Count);

        foreach(OcrQuadRegion detRegion in detRegions)
            overlayBoxes.Add(CreateOverlayBox(detRegion, new SKColor(0, 190, 80, 72), string.Empty, fill: true));

        if(drawBoxDetection)
        {
            foreach(YoloBox box in boxDetections)
            {
                string className = resources.ModelBox.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(box, letterboxInfo, BoxPainter.ClassColorSkia(box.Class), FormatDetectionLabel(className, box.Score)));
            }
        }

        if(drawObbDetection)
        {
            foreach(YoloObb box in obbDetections)
            {
                OcrQuadRegion sourceRegion = YoloObbOcrRegionMapper.MapToSourceRegion(box, mapper);
                string className = resources.ModelObb.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(sourceRegion, SKColors.Red, FormatDetectionLabel(className, box.Score)));
            }
        }

        // OCR text is intentionally kept in the ROI/result panel only. On the main image it hides small boxes
        // and makes class/score labels harder to read during detector tuning.
        return new FrameOverlaySnapshot(frame.Width, frame.Height, overlayBoxes, []);
    }

    static string FormatDetectionLabel(string className, float score)
    {
        string compactClassName = className.Length <= 8 ? className : className[..8];
        return $"{compactClassName} {score * 100:F0}%";
    }

    static OverlayObb CreateOverlayBox(YoloBox box, LetterboxInfo letterboxInfo, SKColor color, string label)
    {
        Point2f topLeft = letterboxInfo.MapBack(box.X, box.Y);
        Point2f bottomRight = letterboxInfo.MapBack(box.W, box.H);
        float left = Math.Min(topLeft.X, bottomRight.X);
        float top = Math.Min(topLeft.Y, bottomRight.Y);
        float right = Math.Max(topLeft.X, bottomRight.X);
        float bottom = Math.Max(topLeft.Y, bottomRight.Y);

        SKPoint[] points =
        [
            new(left, top),
            new(right, top),
            new(right, bottom),
            new(left, bottom),
        ];

        return new OverlayObb(
            new SKPoint((left + right) * 0.5f, (top + bottom) * 0.5f),
            points,
            color,
            label);
    }

    static OverlayObb CreateOverlayBox(OcrQuadRegion sourceRegion, SKColor color, string label, bool fill = false)
    {
        Point2f[] points =
        [
            sourceRegion.Point0,
            sourceRegion.Point1,
            sourceRegion.Point2,
            sourceRegion.Point3,
        ];

        Rect bounds = GetBoundingRect(sourceRegion);
        return new OverlayObb(
            new SKPoint(bounds.X + bounds.Width * 0.5f, bounds.Y + bounds.Height * 0.5f),
            points.Select(point => new SKPoint(point.X, point.Y)).ToArray(),
            color,
            label,
            fill);
    }

    #endregion

    #region Additional Inference Visualization

    static void DrawAdditionalInferenceResults(
        Mat target,
        IBatchedResult? segmentationResult,
        IBatchedResult? classificationResult,
        IDetectionResult<YoloPose>? poseResult,
        LetterboxInfo letterboxInfo,
        InferenceResources resources,
        bool drawSegmentation,
        bool drawClassification,
        bool drawPose)
    {
        if(drawSegmentation && segmentationResult is not null)
            DrawSegmentationResult(target, segmentationResult, letterboxInfo, resources);

        if(drawPose && poseResult is not null)
        {
            YoloPosePainter.DrawPose(
                target,
                poseResult.GetBatch(0).ToArray(),
                letterboxInfo,
                1f,
                1f,
                resources.ModelPose.GetYoloClassName);
        }

        if(drawClassification && classificationResult is YoloCls classification)
            DrawCompactClassification(target, classification, resources.ModelCls.GetYoloClassName);
    }

    static void DrawSegmentationResult(
        Mat target,
        IBatchedResult segmentationResult,
        LetterboxInfo letterboxInfo,
        InferenceResources resources)
    {
        YoloSegResult_FP32_Mask32 fp32Result = segmentationResult switch
        {
            YoloSegResult_FP32_Mask32 item => item,
            YoloSegResult_FP16_Mask32 item => ConvertSegmentationToFP32(item),
            _ => throw new NotSupportedException($"Unsupported SEG result type: {segmentationResult.GetType().Name}")
        };

        SegPainter.DrawSeg(
            target,
            fp32Result.Values,
            fp32Result.Masks,
            letterboxInfo,
            1f,
            1f,
            resources.ModelSeg.GetYoloClassName);
    }

    static void DrawCompactClassification(Mat target, YoloCls result, Func<int, string>? nameResolver)
    {
        string className = nameResolver?.Invoke(result.ClassId) ?? $"Class #{result.ClassId}";
        string label = $"CLS {className} {result.Score:P1}";
        Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.65, 1, out int baseline);
        int panelWidth = Math.Min(target.Width - 16, textSize.Width + 24);
        var panelRect = new Rect(8, 8, panelWidth, textSize.Height + baseline + 18);

        using var overlay = target.Clone();
        Cv2.Rectangle(overlay, panelRect, new Scalar(30, 36, 48), -1, LineTypes.AntiAlias);
        Cv2.AddWeighted(overlay, 0.68, target, 0.32, 0, target);
        Cv2.Rectangle(target, panelRect, new Scalar(40, 220, 150), 1, LineTypes.AntiAlias);
        Cv2.PutText(
            target,
            label,
            new Point(panelRect.X + 12, panelRect.Y + textSize.Height + 8),
            HersheyFonts.HersheySimplex,
            0.65,
            new Scalar(80, 240, 180),
            1,
            LineTypes.AntiAlias);
    }

    static YoloSegResult_FP32_Mask32 ConvertSegmentationToFP32(YoloSegResult_FP16_Mask32 result)
    {
        var values = new YoloSeg_FP32_XYWHSC_Mask32[result.Values.Length];
        for(int index = 0; index < result.Values.Length; index++)
        {
            YoloSeg_FP16_XYWHSC_Mask32 item = result.Values[index];
            var maskCoefficients = new InlineArray_FP32_Mask32();
            for(int maskIndex = 0; maskIndex < InlineArray_FP16_Mask_Count32.Length; maskIndex++)
                maskCoefficients[maskIndex] = (float)item.MaskCoefficients[maskIndex];

            values[index] = new YoloSeg_FP32_XYWHSC_Mask32(
                (float)item.X,
                (float)item.Y,
                (float)item.W,
                (float)item.H,
                (float)item.Score,
                (float)item.ClassId,
                maskCoefficients);
        }

        var masks = new float[result.Masks.Length][];
        for(int index = 0; index < result.Masks.Length; index++)
        {
            Half[] sourceMask = result.Masks[index];
            float[] targetMask = new float[sourceMask.Length];
            for(int maskIndex = 0; maskIndex < sourceMask.Length; maskIndex++)
                targetMask[maskIndex] = (float)sourceMask[maskIndex];

            masks[index] = targetMask;
        }

        return new YoloSegResult_FP32_Mask32
        {
            Values = values,
            Masks = masks,
            PrototypeShape = result.PrototypeShape,
        };
    }

    #endregion

    #region Camera Only Mode

    RealTimeOneFrameData BuildCameraOnlyUpdate(Mat resizedFrame, ref long lastFrameTicks)
    {
        Mat displayFrame = ToBgra(resizedFrame);
        Mat detDisplayFrame = CreateEmptyDetPreview(resizedFrame);
        var metrics = new RealTimeMetricsSnapshot(CalculateFps(ref lastFrameTicks), 0, 0, 0, 0);
        return new RealTimeOneFrameData(displayFrame, detDisplayFrame, new FrameOverlaySnapshot(displayFrame.Width, displayFrame.Height, [], []), [], metrics);
    }

    #endregion

    #region Frame Conversion

    static Mat CaptureFrame(VideoCapture capture, Mat reusableFrame)
    {
        capture.Read(reusableFrame);
        if(!reusableFrame.Empty())
            return reusableFrame.Clone();

        var dummy = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(dummy, "NO SIGNAL", new Point(480, 360), HersheyFonts.HersheySimplex, 1.5, new Scalar(70, 70, 70), 2);
        return dummy;
    }

    Mat CaptureOrReplayFrame(VideoCapture capture, Mat reusableFrame, Mat heldFrame)
    {
        bool shouldReadFrame;
        lock(playbackSyncRoot)
        {
            shouldReadFrame = !isPaused || pendingStepCount > 0 || heldFrame.Empty();
            if(isPaused && pendingStepCount > 0)
                pendingStepCount--;
        }

        if(shouldReadFrame)
        {
            Mat capturedFrame = CaptureFrame(capture, reusableFrame);
            capturedFrame.CopyTo(heldFrame);
            return capturedFrame;
        }

        return heldFrame.Clone();
    }

    static Mat ResizeFrame(Mat input, int targetWidth)
    {
        double aspectRatio = input.Height / (double)input.Width;
        int targetHeight = Math.Max(1, (int)(targetWidth * aspectRatio));
        var resizedMat = new Mat();
        Cv2.Resize(input, resizedMat, new Size(targetWidth, targetHeight));
        return resizedMat;
    }

    static Mat ToBgra(Mat source)
    {
        var bgra = new Mat();
        if(source.Channels() == 4)
            source.CopyTo(bgra);
        else if(source.Channels() == 3)
            Cv2.CvtColor(source, bgra, ColorConversionCodes.BGR2BGRA);
        else if(source.Channels() == 1)
            Cv2.CvtColor(source, bgra, ColorConversionCodes.GRAY2BGRA);
        else
            source.CopyTo(bgra);

        return bgra;
    }

    static Mat CreateDetPreview(Mat scoreMap)
    {
        using var preview8u = new Mat();
        scoreMap.ConvertTo(preview8u, MatType.CV_8UC1, 255.0);
        return ToBgra(preview8u);
    }

    static Mat CreateEmptyDetPreview(Mat sourceFrame)
    {
        using var preview = new Mat(sourceFrame.Height, sourceFrame.Width, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(
            preview,
            "PaddleOCR Det",
            new Point(Math.Max(12, sourceFrame.Width / 2 - 150), Math.Max(32, sourceFrame.Height / 2)),
            HersheyFonts.HersheySimplex,
            0.9,
            new Scalar(70, 70, 70),
            2,
            LineTypes.AntiAlias);
        return ToBgra(preview);
    }

    #endregion

    #region Metrics

    static double ElapsedMilliseconds(System.Diagnostics.Stopwatch stopwatch, long startTicks) =>
        (stopwatch.ElapsedTicks - startTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    static double CalculateFps(ref long lastFrameTicks)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        long elapsedTicks = Math.Max(1, now - lastFrameTicks);
        lastFrameTicks = now;
        return System.Diagnostics.Stopwatch.Frequency / (double)elapsedTicks;
    }

    #endregion
}
