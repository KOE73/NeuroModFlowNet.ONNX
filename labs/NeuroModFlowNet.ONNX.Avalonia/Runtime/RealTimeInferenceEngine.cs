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
    readonly object lifecycleSyncRoot = new();

    CancellationTokenSource? cancellationTokenSource;
    Task? workerTask;
    bool disposed;

    #endregion

    #region Lifecycle

    public RealTimeInferenceEngine(RealTimeAvaloniaSettings settings, RecognitionOptions recognitionOptions)
    {
        this.settings = settings;
        this.recognitionOptions = recognitionOptions;
    }

    public event EventHandler<RealTimeOneFrameData>? FrameReady;
    public event EventHandler<string>? StatusChanged;
    public event EventHandler<IReadOnlyList<RuntimeModelInfo>>? ModelInfoChanged;

    public bool IsRunning => workerTask is { IsCompleted: false };

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
        StatusChanged?.Invoke(this, "Stopped");
    }

    #endregion

    #region Capture Loop

    async Task RunAsync(CancellationToken cancellationToken)
    {
        using VideoCapture capture = VideoCaptureConfig.CreateFromConfig();
        using var sourceFrame = new Mat();
        using InferenceResources? resources = await TryCreateInferenceResourcesAsync(cancellationToken);

        int initializedBatchSize = recognitionOptions.BatchSize;
        int initializedRecognitionShapeVersion = recognitionOptions.RecognitionShapeVersion;
        long lastFrameTicks = System.Diagnostics.Stopwatch.GetTimestamp();

        while(!cancellationToken.IsCancellationRequested)
        {
            using Mat capturedFrame = CaptureFrame(capture, sourceFrame);
            using Mat resizedFrame = ResizeFrame(capturedFrame, recognitionOptions.FrameWidth);

            RealTimeOneFrameData update = resources is null
                ? BuildCameraOnlyUpdate(resizedFrame, ref lastFrameTicks)
                : ProcessInferenceFrame(resources, resizedFrame, ref initializedBatchSize, ref initializedRecognitionShapeVersion, ref lastFrameTicks);

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
            InferenceResources resources = await InferenceResources.CreateAsync(settings, recognitionOptions);
            ModelInfoChanged?.Invoke(this, resources.ModelInfos);
            cancellationToken.ThrowIfCancellationRequested();
            StatusChanged?.Invoke(this, "Running");
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
            return BuildCameraOnlyUpdate(resizedFrame, ref lastFrameTicks);

        var metricsStopwatch = System.Diagnostics.Stopwatch.StartNew();
        using Mat letterboxed = resizedFrame.Letterbox(settings.InputSize, settings.InputSize, out LetterboxInfo letterboxInfo);

        long detectionStartTicks = metricsStopwatch.ElapsedTicks;
        IDetectionResult<YoloBox>? boxResult = boxDetectionEnabled ? resources.RunnerBox.Predict(letterboxed) : null;
        IDetectionResult<YoloObb>? obbResult = obbPredictionNeeded ? resources.RunnerObb.Predict(letterboxed) : null;
        IBatchedResult? segmentationResult = segmentationEnabled ? resources.RunnerSeg.Predict(letterboxed) : null;
        IBatchedResult? classificationResult = classificationEnabled ? resources.RunnerCls.Predict(letterboxed) : null;
        IDetectionResult<YoloPose>? poseResult = poseEnabled ? resources.RunnerPose.Predict(letterboxed) : null;
        double detectionMilliseconds = ElapsedMilliseconds(metricsStopwatch, detectionStartTicks);

        try
        {
            YoloBox[] boxBoxes = boxResult?.GetBatch(0).ToArray() ?? [];
            YoloObb[] obbBoxes = obbResult?.GetBatch(0).ToArray() ?? [];

            long roiStartTicks = metricsStopwatch.ElapsedTicks;
            List<(Mat Roi, Point LabelPoint)> preparedImages = ocrEnabled
                ? PrepareRecognitionImages(resizedFrame, obbBoxes, letterboxInfo)
                : [];
            double roiMilliseconds = ElapsedMilliseconds(metricsStopwatch, roiStartTicks);

            try
            {
                long recognitionStartTicks = metricsStopwatch.ElapsedTicks;
                List<(string Text, Point LabelPoint, Mat Roi)> recognitionRows = ocrEnabled
                    ? PredictRecognition(resources, preparedImages)
                    : [];
                double recognitionMilliseconds = ElapsedMilliseconds(metricsStopwatch, recognitionStartTicks);

                using Mat visualizedFrame = resizedFrame.Clone();
                DrawAdditionalInferenceResults(
                    visualizedFrame,
                    segmentationResult,
                    classificationResult,
                    poseResult,
                    letterboxInfo,
                    resources,
                    segmentationEnabled,
                    classificationEnabled,
                    poseEnabled);
                Mat displayFrame = ToBgra(visualizedFrame);

                FrameOverlaySnapshot overlay = BuildOverlay(
                    resizedFrame,
                    boxBoxes,
                    obbBoxes,
                    letterboxInfo,
                    resources,
                    recognitionRows,
                    boxDetectionEnabled,
                    ocrEnabled || obbDetectionEnabled);

                return new RealTimeOneFrameData(
                    displayFrame,
                    overlay,
                    recognitionRows
                        .Take(12)
                        .Select(row => new RealTimeRecognitionItemData(row.Roi.Clone(), row.Text, recognitionOptions.RoiDisplayScale))
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
                foreach((Mat roi, _) in preparedImages)
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
        }
    }

    #endregion

    #region Recognition ROI Preparation

    List<(Mat Roi, Point LabelPoint)> PrepareRecognitionImages(
        Mat sourceMat,
        ReadOnlySpan<YoloObb> boxes,
        LetterboxInfo letterboxInfo)
    {
        var transform = new ImageResizeTransform(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);

            List<RotatedRect> sourceRects = ImageCoordinateMapper.MapYoloObbToSourceRects(
            boxes,
            transform,
            recognitionOptions.RoiHeightScale);

        List<(Mat Roi, Point LabelPoint)> preparedImages = [];

        foreach(RotatedRect sourceRect in sourceRects)
        {
            Mat? recognitionRoi = OcrRoiExtractor.TryExtractRecognitionRoi(
                sourceMat,
                sourceRect,
                recognitionOptions.RecognitionInputWidth,
                recognitionOptions.RecognitionInputHeight,
                recognitionOptions.ProcessingStage);

            if(recognitionRoi is null) continue;

            preparedImages.Add((recognitionRoi, GetRecognitionLabelPoint(sourceRect, sourceMat)));
        }

        return preparedImages;
    }

    static Point GetRecognitionLabelPoint(RotatedRect sourceRect, Mat sourceMat)
    {
        Rect bounds = ClipRect(sourceRect.BoundingRect(), sourceMat.Width, sourceMat.Height);
        return new Point(bounds.X, Math.Max(0, bounds.Y - RecognitionLabelOffsetY));
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

    List<(string Text, Point LabelPoint, Mat Roi)> PredictRecognition(
        InferenceResources resources,
        List<(Mat Roi, Point LabelPoint)> preparedImages)
    {
        List<(string Text, Point LabelPoint, Mat Roi)> outputResults = [];

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
                (Mat roi, Point labelPoint) = preparedImages[offset + index];
                outputResults.Add((recognitionText, labelPoint, roi));
            }
        }

        return outputResults;
    }

    #endregion

    #region Overlay Preparation

    FrameOverlaySnapshot BuildOverlay(
        Mat frame,
        ReadOnlySpan<YoloBox> boxDetections,
        ReadOnlySpan<YoloObb> obbDetections,
        LetterboxInfo letterboxInfo,
        InferenceResources resources,
        IReadOnlyList<(string Text, Point LabelPoint, Mat Roi)> recognitionRows,
        bool drawBoxDetection,
        bool drawObbDetection)
    {
        var transform = new ImageResizeTransform(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);
        List<OverlayObb> overlayBoxes = new(boxDetections.Length + obbDetections.Length);

        if(drawBoxDetection)
        {
            foreach(YoloBox box in boxDetections)
            {
                string className = resources.ModelBox.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(box, letterboxInfo, BoxPainter.ClassColorSkia(box.Class), $"{className} {box.Score:P0}"));
            }
        }

        if(drawObbDetection)
        {
            foreach(YoloObb box in obbDetections)
            {
                RotatedRect sourceRect = ImageCoordinateMapper.MapYoloObbToSourceRect(box, transform);
                string className = resources.ModelObb.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(sourceRect, BoxPainter.ClassColorSkia(box.Class), $"{className} {box.Score:P0}"));
            }
        }

        OverlayText[] texts = recognitionRows
            .Select(row => new OverlayText(new SKPoint(row.LabelPoint.X, row.LabelPoint.Y), row.Text))
            .ToArray();

        return new FrameOverlaySnapshot(frame.Width, frame.Height, overlayBoxes, texts);
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

    static OverlayObb CreateOverlayBox(RotatedRect sourceRect, SKColor color, string label)
    {
        Point2f[] points = sourceRect.Points();
        return new OverlayObb(
            new SKPoint(sourceRect.Center.X, sourceRect.Center.Y),
            points.Select(point => new SKPoint(point.X, point.Y)).ToArray(),
            color,
            label);
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
        var metrics = new RealTimeMetricsSnapshot(CalculateFps(ref lastFrameTicks), 0, 0, 0, 0);
        return new RealTimeOneFrameData(displayFrame, new FrameOverlaySnapshot(displayFrame.Width, displayFrame.Height, [], []), [], metrics);
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
        Cv2.CvtColor(source, bgra, ColorConversionCodes.BGR2BGRA);
        return bgra;
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
