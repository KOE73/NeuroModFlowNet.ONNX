using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Diagnostics;

using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;
using Spectre.Console;
using Spectre.Console.Rendering;


namespace OnnxTestLoader;

/// <summary>
/// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-12x
/// </summary>
// CODEX: Keep model*/runner*/result* names padded with trailing underscores so aligned batches read like tables.
public class RealTimeView2 : IDisposable
{
    #region Const and Fields

    const float RecognitionRoiHeightScale = 2f;
    const int RecognitionInputHeight = 48;
    const int RecognitionInputWidth = 640;
    const int RecognitionOutputItemCount = RecognitionInputWidth / PaddleOCRRecExtractor.OutputWidthStride;
    const int RecognitionBatchMin = 1;
    const int RecognitionBatchMax = 16;
    const int RecognitionMaxRegionsPerFrame = 64;
    const bool RecognitionOverlapSuppressionEnabled = true;
    const bool RecognitionLineMergeEnabled = false;
    const int RecognitionLabelOffsetY = 6;
    const double RecognitionRoiBrightness = 40;
    const double RecognitionRoiContrastPercent = 115;
    const double RecognitionRoiGamma = 5.0;
    const double RecognitionRoiBrightnessStep = 5;
    const double RecognitionRoiContrastStep = 5;
    const double RecognitionRoiGammaStep = 0.25;

    bool showSeg = false;
    int windowOutW = 640;
    int recognitionDisplayMode = 1;
    int recognitionBatchSize = 1;
    bool recognitionScreenOutputEnabled = true;
    bool recognitionWindowOutputEnabled = true;
    bool recognitionRoiProcessingEnabled = true;
    bool disposed;

    readonly RealTimeViewSettings _settings = RealTimeViewSettings.FromConfig();
    readonly TextRegionBrightnessContrastStage recognitionRoiBrightnessContrastStage = new()
    {
        Brightness = RecognitionRoiBrightness,
        ContrastPercent = RecognitionRoiContrastPercent
    };
    readonly TextRegionGammaCorrectionStage recognitionRoiGammaCorrectionStage = new()
    {
        Gamma = RecognitionRoiGamma
    };
    readonly TextRegionProcessingPipeline recognitionRoiProcessingStage;

    VideoCapture? capture;
    Mat? sourceMat;
    Mat? bgraMat;

    #region Models and Runners

    OnnxRuntimeContext? modelBox_;
    OnnxRuntimeContext? modelObb_;
    OnnxRuntimeContext? modelPose;
    OnnxRuntimeContext? modelSeg_;
    OnnxRuntimeContext? modelDv3_;
    OnnxRuntimeContext? modelDv5_;
    OnnxRuntimeContext? modelRec_;

    IRunner<Mat, IDetectionResult<YoloBox>>? runnerBox_;
    IRunner<Mat, IDetectionResult<YoloObb>>? runnerObb_;
    IRunner<Mat, IDetectionResult<YoloPose>>? runnerPose;
    IRunner<Mat, Mat>? runnerDv3_;
    IRunner<Mat, Mat>? runnerDv5_;
    ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>? runnerRec_;

    #endregion

    #endregion

    #region Lifecycle

    public RealTimeView2()
    {
        recognitionRoiProcessingStage = new TextRegionProcessingPipeline(
            recognitionRoiBrightnessContrastStage,
            recognitionRoiGammaCorrectionStage);
    }

    public void Dispose()
    {
        if(disposed) return;

        disposed = true;
        DisposeResources();
        recognitionRoiProcessingStage.Dispose();
        GC.SuppressFinalize(this);
    }

    private void DisposeResources()
    {
        runnerBox_?.Dispose();
        runnerObb_?.Dispose();
        runnerPose?.Dispose();
        runnerDv3_?.Dispose();
        runnerDv5_?.Dispose();
        runnerRec_?.Dispose();

        modelSeg_?.Dispose();

        capture?.Dispose();
        window?.Dispose();
        windowPose?.Dispose();
        sourceMat?.Dispose();
        bgraMat?.Dispose();

        runnerBox_ = null; runnerObb_ = null; runnerPose = null; runnerDv3_ = null; runnerDv5_ = null; runnerRec_ = null;
        modelBox_ = null; modelObb_ = null; modelPose = null; modelSeg_ = null; modelDv3_ = null; modelDv5_ = null; modelRec_ = null;

        capture = null;
        window = null;
        windowPose = null;
        sourceMat = null;
        bgraMat = null;
        fpsMonitor = null;
        recognitionRows.Clear();
    }

    #endregion

    #region Public API

    public async void ForAI()
    {
        string modelBoxPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.BoxModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));
        string modelObbPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.ObbModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));

        var localModelBox = new OnnxRuntimeContext(modelBoxPath);
        var localModelObb = new OnnxRuntimeContext(modelObbPath);

        using var localRunnerBox = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(localModelBox);
        using var localRunnerObb = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(localModelObb);
    }

    public async Task Run()
    {
        ThrowIfDisposed();

        await InitializeAsync();
        StartLive(LiveLoop);
    }

    #endregion

    #region Initialization

    private async Task InitializeAsync()
    {
        DisposeResources();

        string modelBoxPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.BoxModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));
        string modelObbPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.ObbModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));
        string modelPosePath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.PoseModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));
        string modelSegPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(_settings.SegModelName, _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.UseByteBgr));

        string modelDv3Path = await AssetsManager.GetAssetPathAsync(GetPaddleModelPath("/paddleocr/detection/v3/det.onnx", _settings.PaddleDetModelPrecision, _settings.PaddleDetUseByteBgr));
        string modelDv5Path = await AssetsManager.GetAssetPathAsync(GetPaddleModelPath("/paddleocr/detection/v5/det.onnx", _settings.PaddleDetModelPrecision, _settings.PaddleDetUseByteBgr));

        string recognitionModelPath = await AssetsManager.GetAssetPathAsync(GetPaddleModelPath("/paddleocr/languages/english/rec.onnx", _settings.PaddleRecModelPrecision, _settings.PaddleRecUseByteBgr));
        // PaddleOCRRecExtractor loads dict.txt from the recognition model folder during initialization.
        await AssetsManager.GetAssetPathAsync("/paddleocr/languages/english/dict.txt");

        modelBox_ = new OnnxRuntimeContext(modelBoxPath, _settings.InferenceBackend);
        modelObb_ = new OnnxRuntimeContext(modelObbPath, _settings.InferenceBackend);
        modelPose = new OnnxRuntimeContext(modelPosePath, _settings.InferenceBackend);
        modelSeg_ = new OnnxRuntimeContext(modelSegPath, _settings.InferenceBackend);
        modelDv3_ = new OnnxRuntimeContext(modelDv3Path, _settings.PaddleDetInferenceBackend);
        modelDv5_ = new OnnxRuntimeContext(modelDv5Path, _settings.PaddleDetInferenceBackend);
        modelRec_ = new OnnxRuntimeContext(recognitionModelPath, _settings.PaddleRecInferenceBackend);

        InitRecognitionModelBatch(recognitionBatchSize);

        WriteModelInfo();
        InitializeRunners();
        WriteRecognitionInfo();
        InitializeVideoAndDisplay();

        sourceMat = new Mat();
        bgraMat = new Mat();
        fpsMonitor = new FpsConsoleMonitor();
        lastTick = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private void WriteModelInfo()
    {
        modelBox_!.WriteInfo();
        modelObb_!.WriteInfo();
        modelPose!.WriteInfo();
        modelSeg_!.WriteInfo();
        modelDv3_!.WriteInfo();
        modelDv5_!.WriteInfo();
        modelRec_!.WriteInfo();
    }

    private void InitializeRunners()
    {
        runnerBox_ = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(modelBox_!);
        runnerObb_ = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(modelObb_!);
        runnerPose = YoloPoseFactory.CreateRunner(modelPose!);

        runnerObb_.OutAs<IExtractorThreshold>()?.Threshold = 0.3f;

        runnerDv3_ = PaddleOCRDetFactory.CreateRunner<Mat, Mat>(modelDv3_!, MatType.CV_8UC1);
        runnerDv5_ = PaddleOCRDetFactory.CreateRunner<Mat, Mat>(modelDv5_!, MatType.CV_8UC1);
        runnerRec_ = new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(modelRec_!);
    }

    private void WriteRecognitionInfo()
    {
        PaddleOCRRecExtractor? recognitionExtractor = runnerRec_?.OutAs<PaddleOCRRecExtractor>();
        if(recognitionExtractor?.IsAlphabetLoaded == true)
            AnsiConsole.MarkupLine($"[cyan]PaddleOCR Rec dictionary characters:[/] [yellow]{recognitionExtractor.Alphabet.Length - 2}[/]");
    }

    private void InitRecognitionModelBatch(int batchSize)
    {
        modelRec_!.InitInputPersistentValue(modelRec_.PrimaryInputName, [batchSize, 3, RecognitionInputHeight, RecognitionInputWidth]);
        modelRec_.InitOutputPersistentValue(modelRec_.PrimaryOutputName, [batchSize, RecognitionOutputItemCount, 438]);
    }

    private void SetRecognitionBatchSize(int batchSize)
    {
        batchSize = Math.Clamp(batchSize, RecognitionBatchMin, RecognitionBatchMax);
        if(batchSize == recognitionBatchSize) return;

        recognitionBatchSize = batchSize;
        InitRecognitionModelBatch(recognitionBatchSize);
        runnerRec_ = new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(modelRec_!);
    }

    private static string GetPaddleModelPath(string modelPath, string precision, bool isByteBgr)
    {
        if(string.Equals(precision, "fp32", StringComparison.OrdinalIgnoreCase) && !isByteBgr)
            return modelPath;

        string modelDirectory = Path.GetDirectoryName(modelPath)?.Replace('\\', '/') ?? string.Empty;
        string modelFileName = Path.GetFileNameWithoutExtension(modelPath);
        string modelExtension = Path.GetExtension(modelPath);
        string precisionSuffix = string.Equals(precision, "fp32", StringComparison.OrdinalIgnoreCase) ? string.Empty : $"_{precision}";
        string byteBgrSuffix = isByteBgr ? "_bytebgr" : string.Empty;
        string configuredModelFileName = $"{modelFileName}{precisionSuffix}{byteBgrSuffix}{modelExtension}";

        return string.IsNullOrEmpty(modelDirectory)
            ? configuredModelFileName
            : $"{modelDirectory}/{configuredModelFileName}";
    }

    private void InitializeVideoAndDisplay()
    {
        capture = VideoCaptureConfig.CreateFromConfig();

        if(!capture.IsOpened())
            AnsiConsole.MarkupLine("[yellow]Warning: Camera not found — simulation (black frame) mode.[/]");

        window = new Window("AI DASHBOARD [NeuroModFlowNet.ONNX]");
        windowPose = new Window("AI DASHBOARD [NeuroModFlowNet.ONNX]");
    }

    #endregion

    #region Live Loop

    private void LiveLoop(LiveDisplayContext liveContext)
    {
        while(true)
        {
            CaptureFrame();

            if(!KeyHandlingOrQuit()) break;

            using var resizedMat = ResizeFrame(sourceMat!);
            using var letterboxed = resizedMat.Letterbox(_settings.InputSize, _settings.InputSize, out var info);

            InitDetModels(letterboxed);

            long detPredictionStartTicks = detPredictionMetric.Start();
            IDetectionResult<YoloBox> resultBox_ = runnerBox_!.Predict(letterboxed);
            IDetectionResult<YoloObb> resultObb_ = runnerObb_!.Predict(letterboxed);
            IDetectionResult<YoloPose> resultPose = runnerPose!.Predict(letterboxed);
            using Mat resultDv3_ = runnerDv3_!.Predict(letterboxed);
            using Mat resultDv5_ = runnerDv5_!.Predict(letterboxed);
            detPredictionMetric.AddElapsed(detPredictionStartTicks);

            YoloSegResult_FP16_Mask32 resultSeg_ = default!;

            Cv2.CvtColor(resizedMat, bgraMat!, ColorConversionCodes.BGR2BGRA);

            BoxPainter.DrawBoxSkia(bgraMat!, resultBox_.GetBatch(0).ToArray(), info, 1f, 1f, modelBox_!.GetYoloClassName);
            ObbPainter.DrawObb(bgraMat!, resultObb_.GetBatch(0).ToArray(), info, 1f, 1f, nameResolver: modelObb_!.GetYoloClassName);
            YoloPosePainter.DrawPose(bgraMat!, resultPose.GetBatch(0).ToArray(), info, 1f, 1f, modelPose!.GetYoloClassName);

            if(showSeg)
            {
                // resultSeg_ = runnerSeg.Predict(letterboxed);
                //
                // YoloSeg_FP16_XYWHSC_Mask32[] dets = resultSeg_.Values;
                // Half[][] masks = resultSeg_.Masks;
                // SegPainter.DrawSeg(bgraMat, dets, masks, info, 1f, 1f, modelSeg_.GetYoloClassName);
            }

            recognitionRows.Clear();
            ProcessRecognition(resizedMat, bgraMat!, resultObb_.GetBatch(0), info);
            ShowFrames(resultDv3_, resultDv5_);
            UpdateFps(liveContext);

            (resultBox_ as IDisposable)?.Dispose();
            (resultObb_ as IDisposable)?.Dispose();
            (resultPose as IDisposable)?.Dispose();
            (resultSeg_ as IDisposable)?.Dispose();
        }
    }

     void CaptureFrame()
    {
        capture!.Read(sourceMat!);
        if(!sourceMat!.Empty()) return;

        using var dummy = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(dummy, "NO SIGNAL", new Point(480, 360),
                            HersheyFonts.HersheySimplex, 1.5, new Scalar(60, 60, 60), 2);
        dummy.CopyTo(sourceMat);
    }

     Mat ResizeFrame(Mat input)
    {
        double aspectRatio = (double)input.Height / input.Width;
        int targetHeight = (int)(windowOutW * aspectRatio);
        var resizedMat = new Mat();
        Cv2.Resize(input, resizedMat, new OpenCvSharp.Size(windowOutW, targetHeight));
        return resizedMat;
    }

    #endregion

    #region OpenCV Window Display

    const int RecognitionDebugWindowsMax = 10;
    const int RecognitionDebugWindowX = 0;
    const int RecognitionDebugWindowY = 40;
    const int RecognitionDebugWindowStepY = RecognitionInputHeight + 32;

    Window? window;
    Window? windowPose;

     void ShowFrames(Mat resultDv3_, Mat resultDv5_)
    {
        window!.ShowImage(bgraMat!);

        Cv2.ImShow("resultDv3", resultDv3_);
        Cv2.ImShow("resultDv5", resultDv5_);
    }

    private static void ShowRecognitionWindowResults(List<(Mat Roi, Point LabelPoint)> preparedImages)
    {
        int debugWindowCount = Math.Min(RecognitionDebugWindowsMax, preparedImages.Count);

        for(int index = 0; index < debugWindowCount; index++)
        {
            ShowRecognitionDebugWindow(preparedImages[index].Roi, index);
        }
    }

    private static void ShowRecognitionDebugWindow(Mat recognitionRoi, int index)
    {
        string windowName = $"ocr_roi_{index + 1:00}";
        Cv2.ImShow(windowName, recognitionRoi);
        Cv2.MoveWindow(
            windowName,
            RecognitionDebugWindowX,
            RecognitionDebugWindowY + index * RecognitionDebugWindowStepY);
    }

    #endregion

    #region Spectre Live Display

    const int RecognitionRowsMax = 12;
    const int RecognitionPanelWidth = 70;
    const double MetricsAverageWindowSeconds = 3;

    long lastTick;

    readonly List<string> recognitionRows = [];
    readonly TimedStageMetric detPredictionMetric = new(TimeSpan.FromSeconds(MetricsAverageWindowSeconds));
    readonly TimedStageMetric recPreparationMetric = new(TimeSpan.FromSeconds(MetricsAverageWindowSeconds));
    readonly TimedStageMetric recPredictionTotalMetric = new(TimeSpan.FromSeconds(MetricsAverageWindowSeconds));
    readonly TimedStageMetric recPredictionCallMetric = new(TimeSpan.FromSeconds(MetricsAverageWindowSeconds));

    FpsConsoleMonitor? fpsMonitor;

    private void StartLive(Action<LiveDisplayContext> action)
    {
        AnsiConsole.Live(BuildLiveScreen())
            .Overflow(VerticalOverflow.Crop)
            .Cropping(VerticalOverflowCropping.Top)
            .AutoClear(false)
            .Start(action);
    }

    private void UpdateFps(LiveDisplayContext liveContext)
    {
        long elapsedTicks = System.Diagnostics.Stopwatch.GetTimestamp() - lastTick;
        double fps = System.Diagnostics.Stopwatch.Frequency / (double)elapsedTicks;
        lastTick = System.Diagnostics.Stopwatch.GetTimestamp();

        fpsMonitor!.AddFrame(fps);
        if(!fpsMonitor.ShouldRender) return;

        liveContext.UpdateTarget(BuildLiveScreen());
        liveContext.Refresh();
        fpsMonitor.RestartRenderTimer();
    }

    private IRenderable BuildLiveScreen()
    {
        // Left child: FPS and stage timings stay together so the recognition panel remains readable.
        IRenderable leftPanel = new Rows(
            fpsMonitor!.Render(),
            BuildPerformancePanel());

        // Right child: OCR text is intentionally a separate table, making columns, borders and row limits easy to edit.
        IRenderable recognitionPanel = BuildRecognitionPanel();

        // Root layout: this is the only place that decides how child renderables are arranged on the live screen.
        var layoutGrid = new Grid();
        layoutGrid.AddColumn(new GridColumn().PadRight(1));
        layoutGrid.AddColumn(new GridColumn().NoWrap().Width(RecognitionPanelWidth));
        layoutGrid.AddRow(leftPanel, recognitionPanel);

        return new Rows(
            BuildKeyHelpPanel(),
            layoutGrid);
    }

    private IRenderable BuildKeyHelpPanel()
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn(new TableColumn("[grey]Keys[/]").NoWrap())
            .AddColumn(new TableColumn("[grey]Action[/]").NoWrap())
            .AddColumn(new TableColumn("[grey]Current[/]").NoWrap());

        table.AddRow("[cyan]Esc[/]", "exit", string.Empty);
        table.AddRow("[cyan]+/-[/]", "window width", $"[yellow]{windowOutW}[/]");
        table.AddRow("[cyan]Q/W[/]", "recognition batch", $"[yellow]{recognitionBatchSize}[/]");
        table.AddRow("[cyan]1/2/3[/]", "text mode", $"[yellow]{GetRecognitionModeName()}[/]");
        table.AddRow("[cyan]4[/]", "screen overlay", recognitionScreenOutputEnabled ? "[green]on[/]" : "[red]off[/]");
        table.AddRow("[cyan]5[/]", "ROI debug windows", recognitionWindowOutputEnabled ? "[green]on[/]" : "[red]off[/]");
        table.AddRow("[cyan]6[/]", "ROI processing", recognitionRoiProcessingEnabled ? "[green]on[/]" : "[red]off[/]");
        table.AddRow("[cyan]A/Z[/]", "brightness +/-", $"[yellow]{recognitionRoiBrightnessContrastStage.Brightness:F0}[/]");
        table.AddRow("[cyan]S/X[/]", "contrast +/-", $"[yellow]{recognitionRoiBrightnessContrastStage.ContrastPercent:F0}%[/]");
        table.AddRow("[cyan]D/C[/]", "gamma +/-", $"[yellow]{recognitionRoiGammaCorrectionStage.Gamma:F2}[/]");
        table.AddRow("[cyan]R[/]", "rotate processing stages", Markup.Escape(GetRoiProcessingStageList()));

        return new Panel(table)
            .Header("[cyan]Keys[/]")
            .Border(BoxBorder.Rounded)
            .Padding(1, 0);
    }

    private IRenderable BuildPerformancePanel()
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn(new TableColumn("[grey]Stage[/]").NoWrap())
            .AddColumn(new TableColumn("[grey]Now ms[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[grey]Avg {MetricsAverageWindowSeconds:F0}s ms[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[grey]Avg {MetricsAverageWindowSeconds:F0}s calls/s[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[grey]Avg {MetricsAverageWindowSeconds:F0}s items/s[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn("[grey]Batch[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn("[grey]Fill[/]").RightAligned().NoWrap());

        AddMetricRow(table, "Det Predict", detPredictionMetric);
        AddMetricRow(table, "Rec Crop", recPreparationMetric);
        AddMetricRow(table, "Rec Predict total", recPredictionTotalMetric);
        AddMetricRow(table, "Rec Predict call", recPredictionCallMetric, recognitionBatchSize);

        table.AddRow(string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty);
        table.AddRow("[grey]Rec batch Q/W[/]", $"[bold cyan]{recognitionBatchSize}[/]", $"[grey]{RecognitionBatchMin}-{RecognitionBatchMax}[/]", string.Empty, string.Empty, string.Empty, string.Empty);
        table.AddRow("[grey]Screen 4[/]", recognitionScreenOutputEnabled ? "[green]on[/]" : "[red]off[/]", string.Empty, string.Empty, string.Empty, string.Empty, string.Empty);
        table.AddRow("[grey]ROI windows 5[/]", recognitionWindowOutputEnabled ? "[green]on[/]" : "[red]off[/]", string.Empty, string.Empty, string.Empty, string.Empty, string.Empty);
        table.AddRow(
            "[grey]ROI prep 6[/]",
            recognitionRoiProcessingEnabled ? "[green]on[/]" : "[red]off[/]",
            $"[grey]b {recognitionRoiBrightnessContrastStage.Brightness:F0} c {recognitionRoiBrightnessContrastStage.ContrastPercent:F0} g {recognitionRoiGammaCorrectionStage.Gamma:F2}[/]",
            string.Empty,
            string.Empty,
            string.Empty,
            string.Empty);

        return new Panel(table)
            .Header("[cyan]OCR Timing[/]")
            .Border(BoxBorder.Rounded);
    }

    private static void AddMetricRow(Table table, string title, TimedStageMetric metric, int? batchCapacity = null)
    {
        TimedStageMetricSnapshot snapshot = metric.Snapshot();
        double fillPercent = batchCapacity is > 0
            ? snapshot.AverageItemsPerCall / batchCapacity.Value * 100.0
            : 0;

        table.AddRow(
            Markup.Escape(title),
            $"[yellow]{snapshot.CurrentMilliseconds:F2}[/]",
            $"[green]{snapshot.AverageMilliseconds:F2}[/]",
            $"[green]{snapshot.AverageCallFps:F1}[/]",
            $"[bold green]{snapshot.AverageItemFps:F1}[/]",
            snapshot.AverageItemsPerCall <= 0 ? "[grey]-[/]" : $"[yellow]{snapshot.AverageItemsPerCall:F2}[/]",
            batchCapacity is > 0 ? $"[yellow]{fillPercent:F0}%[/]" : "[grey]-[/]");
    }

    private IRenderable BuildRecognitionPanel()
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Width(RecognitionPanelWidth - 4);

        table.AddColumn(new TableColumn("[grey]#[/]").RightAligned().NoWrap());
        table.AddColumn(new TableColumn("[cyan]Recognized text[/]").Width(RecognitionPanelWidth - 12));

        if(recognitionRows.Count == 0)
        {
            table.AddRow("[grey]-[/]", "[grey]waiting...[/]");
        }
        else
        {
            for(int index = 0; index < recognitionRows.Count; index++)
            {
                table.AddRow(
                    $"[grey]{index + 1}[/]",
                    Markup.Escape(recognitionRows[index]));
            }
        }

        return new Panel(table)
            .Header($"[green]OCR[/] [grey](1 standard, 2 spaces, 3 candidates)[/] [yellow]{GetRecognitionModeName()}[/]")
            .Border(BoxBorder.Rounded);
    }

    #endregion

    #region Input And Model State

    private bool KeyHandlingOrQuit()
    {
        int key = Cv2.WaitKey(1);
        if(key == 27) return false;

        if(key == '+' || key == '=') windowOutW = Math.Min(windowOutW + 40, 1920);
        if(key == '-' || key == '_') windowOutW = Math.Max(windowOutW - 40, 320);
        if(key == '1') recognitionDisplayMode = 1;
        if(key == '2') recognitionDisplayMode = 2;
        if(key == '3') recognitionDisplayMode = 3;
        if(key == 'q' || key == 'Q') SetRecognitionBatchSize(recognitionBatchSize - 1);
        if(key == 'w' || key == 'W') SetRecognitionBatchSize(recognitionBatchSize + 1);
        if(key == '4') recognitionScreenOutputEnabled = !recognitionScreenOutputEnabled;
        if(key == '5') recognitionWindowOutputEnabled = !recognitionWindowOutputEnabled;
        if(key == '6') recognitionRoiProcessingEnabled = !recognitionRoiProcessingEnabled;
        if(key == 'a' || key == 'A') AdjustRecognitionRoiBrightness(RecognitionRoiBrightnessStep);
        if(key == 'z' || key == 'Z') AdjustRecognitionRoiBrightness(-RecognitionRoiBrightnessStep);
        if(key == 's' || key == 'S') AdjustRecognitionRoiContrast(RecognitionRoiContrastStep);
        if(key == 'x' || key == 'X') AdjustRecognitionRoiContrast(-RecognitionRoiContrastStep);
        if(key == 'd' || key == 'D') AdjustRecognitionRoiGamma(RecognitionRoiGammaStep);
        if(key == 'c' || key == 'C') AdjustRecognitionRoiGamma(-RecognitionRoiGammaStep);
        if(key == 'r' || key == 'R') RotateRecognitionRoiProcessingStages();
        return true;
    }

    private void AdjustRecognitionRoiBrightness(double delta)
    {
        recognitionRoiBrightnessContrastStage.Brightness =
            Math.Clamp(recognitionRoiBrightnessContrastStage.Brightness + delta, -255, 255);
    }

    private void AdjustRecognitionRoiContrast(double delta)
    {
        recognitionRoiBrightnessContrastStage.ContrastPercent =
            Math.Clamp(recognitionRoiBrightnessContrastStage.ContrastPercent + delta, 0, 300);
    }

    private void AdjustRecognitionRoiGamma(double delta)
    {
        recognitionRoiGammaCorrectionStage.Gamma =
            Math.Clamp(recognitionRoiGammaCorrectionStage.Gamma + delta, 0.1, 10);
    }

    private void RotateRecognitionRoiProcessingStages()
    {
        if(recognitionRoiProcessingStage.Count < 2) return;

        recognitionRoiProcessingStage.Move(0, recognitionRoiProcessingStage.Count - 1);
    }

    private string GetRoiProcessingStageList() =>
        string.Join(" -> ", recognitionRoiProcessingStage.GetStages().Select(stage => stage.Name));

    private void InitDetModels(Mat letterboxed)
    {
        if(!modelDv3_!.IsInputPersistentValueInitialized(modelDv3_.PrimaryInputName))
        {
            modelDv3_.InitInputPersistentValue(modelDv3_.PrimaryInputName, [1, 3, letterboxed.Width, letterboxed.Height]);
            modelDv3_.InitOutputPersistentValue(modelDv3_.PrimaryOutputName, [1, 1, letterboxed.Width, letterboxed.Height]);
        }

        if(!modelDv5_!.IsInputPersistentValueInitialized(modelDv5_.PrimaryInputName))
        {
            modelDv5_.InitInputPersistentValue(modelDv5_.PrimaryInputName, [1, 3, letterboxed.Width, letterboxed.Height]);
            modelDv5_.InitOutputPersistentValue(modelDv5_.PrimaryOutputName, [1, 1, letterboxed.Width, letterboxed.Height]);
        }
    }

    private void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(RealTimeView2));
    }

    #endregion

    #region Recognition

    #region Recognition Pipeline

    private void ProcessRecognition(
        Mat sourceMat,
        Mat targetMat,
        ReadOnlySpan<YoloObb> boxes,
        LetterboxInfo letterboxInfo)
    {
        long preparationStartTicks = recPreparationMetric.Start();
        List<(Mat Roi, Point LabelPoint)> preparedImages = PrepareRecognitionImages(sourceMat, boxes, letterboxInfo);
        recPreparationMetric.AddElapsed(preparationStartTicks, preparedImages.Count);

        try
        {
            long predictionStartTicks = recPredictionTotalMetric.Start();
            List<(string Text, Point LabelPoint, Mat Roi)> recognitionResults = PredictRecognitionBatches(preparedImages);
            recPredictionTotalMetric.AddElapsed(predictionStartTicks, preparedImages.Count);

            WriteRecognitionConsoleResults(recognitionResults);

            if(recognitionScreenOutputEnabled)
                DrawRecognitionScreenResults(targetMat, recognitionResults);

            if(recognitionWindowOutputEnabled)
                ShowRecognitionWindowResults(preparedImages);
        }
        finally
        {
            foreach((Mat roi, _) in preparedImages)
                roi.Dispose();
        }
    }

    #endregion

    #region Recognition ROI Preparation

    private List<(Mat Roi, Point LabelPoint)> PrepareRecognitionImages(
        Mat sourceMat,
        ReadOnlySpan<YoloObb> boxes,
        LetterboxInfo letterboxInfo)
    {
        var mapper = LetterboxCoordinateMapper.Create(
            letterboxInfo.Ratio,
            letterboxInfo.OffsetX,
            letterboxInfo.OffsetY);

        // Most realtime OCR frames have a small number of text boxes. Keep the detector-to-region
        // bridge stack-only; region postprocessing owns the cached geometry for filters/overlap/merge.
        Span<OcrQuadRegion> mappedSourceRegions = boxes.Length <= 1000
            ? stackalloc OcrQuadRegion[boxes.Length]
            : new OcrQuadRegion[boxes.Length];

        YoloObbOcrRegionMapper.MapToSourceRegions(
            boxes,
            mapper,
            RecognitionRoiHeightScale,
            mappedSourceRegions);

        var postprocessOptions = new OcrRegionPostprocessorOptions
        {
            MaxRegions = RecognitionMaxRegionsPerFrame,
            EnableOverlapSuppression = RecognitionOverlapSuppressionEnabled,
            EnableLineMerge = RecognitionLineMergeEnabled,
            MaxMergedRegionAspectRatio = RecognitionInputWidth / (float)RecognitionInputHeight,
        };

        List<OcrQuadRegion> sourceRegions = [];
        OcrRegionPostprocessor.Shared.Process(mappedSourceRegions, postprocessOptions, sourceRegions);

        var options = new TextRegionExtractionOptions(
            RecognitionInputWidth,
            RecognitionInputHeight,
            recognitionRoiProcessingEnabled ? recognitionRoiProcessingStage : null);

        List<(Mat Roi, Point LabelPoint)> preparedImages = [];

        foreach(OcrQuadRegion sourceRegion in sourceRegions)
        {
            if(!NaiveTextRegionExtractor.Shared.TryExtract(
                sourceMat,
                sourceRegion,
                options,
                out Mat? recognitionRoi)) continue;
            if(recognitionRoi is null) continue;

            preparedImages.Add((recognitionRoi, GetRecognitionLabelPoint(sourceRegion, sourceMat)));
        }

        return preparedImages;
    }

    private static Point GetRecognitionLabelPoint(OcrQuadRegion sourceRegion, Mat sourceMat)
    {
        Rect bounds = ClipRect(GetBoundingRect(sourceRegion), sourceMat.Width, sourceMat.Height);
        return new Point(bounds.X, Math.Max(0, bounds.Y - RecognitionLabelOffsetY));
    }

    private static Rect GetBoundingRect(OcrQuadRegion sourceRegion)
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

    private static Rect ClipRect(Rect rect, int width, int height)
    {
        int x = Math.Clamp(rect.X, 0, width - 1);
        int y = Math.Clamp(rect.Y, 0, height - 1);
        int right = Math.Clamp(rect.Right, x + 1, width);
        int bottom = Math.Clamp(rect.Bottom, y + 1, height);
        return new Rect(x, y, right - x, bottom - y);
    }

    #endregion

    #region Recognition Batch Prediction

    private List<(string Text, Point LabelPoint, Mat Roi)> PredictRecognitionBatches(List<(Mat Roi, Point LabelPoint)> preparedImages)
    {
        List<(string Text, Point LabelPoint, Mat Roi)> outputResults = [];

        for(int offset = 0; offset < preparedImages.Count; offset += recognitionBatchSize)
        {
            int itemCount = Math.Min(recognitionBatchSize, preparedImages.Count - offset);
            List<Mat> batchImages = new(itemCount);

            for(int index = 0; index < itemCount; index++)
                batchImages.Add(preparedImages[offset + index].Roi);

            List<PaddleOCRRecExtractor.OcrResult> batchResults;
            try
            {
                batchResults = recPredictionCallMetric.Measure(
                    () => runnerRec_!.Predict(batchImages),
                    batchImages.Count);
            }
            catch(Exception)
            {
                continue;
            }

            int resultCount = Math.Min(itemCount, batchResults.Count);
            for(int index = 0; index < resultCount; index++)
            {
                PaddleOCRRecExtractor.OcrResult recognitionResult = batchResults[index];
                if(recognitionResult.IsEmpty) continue;

                string recognitionText = GetRecognitionText(recognitionResult);
                (Mat roi, Point labelPoint) = preparedImages[offset + index];
                outputResults.Add((recognitionText, labelPoint, roi));
            }
        }

        return outputResults;
    }

    #endregion

    #region Recognition Text Mode

    private string GetRecognitionText(PaddleOCRRecExtractor.OcrResult recognitionResult) =>
        recognitionDisplayMode switch
        {
            1 => recognitionResult.Standard,
            2 => recognitionResult.WithSpaces,
            3 => recognitionResult.FullCandidates,
            _ => recognitionResult.Standard,
        };

    private string GetRecognitionModeName() =>
        recognitionDisplayMode switch
        {
            1 => "mode: standard",
            2 => "mode: spaces",
            3 => "mode: candidates",
            _ => "mode: standard",
        };

    #endregion

    #region Recognition Output

    private void WriteRecognitionConsoleResults(List<(string Text, Point LabelPoint, Mat Roi)> recognitionResults)
    {
        foreach((string text, _, _) in recognitionResults)
            AddRecognitionRow(text);
    }

    private void AddRecognitionRow(string text)
    {
        if(string.IsNullOrWhiteSpace(text)) return;
        if(recognitionRows.Count >= RecognitionRowsMax) return;

        recognitionRows.Add(text);
    }

    private static void DrawRecognitionScreenResults(
        Mat targetMat,
        List<(string Text, Point LabelPoint, Mat Roi)> recognitionResults)
    {
        foreach((string recognitionText, Point labelPoint, _) in recognitionResults)
        {
            try
            {
                Cv2.PutText(
                    targetMat,
                    recognitionText,
                    labelPoint,
                    HersheyFonts.HersheySimplex,
                    0.55,
                    Scalar.LimeGreen,
                    2,
                    LineTypes.AntiAlias);
            }
            catch(Exception)
            {

            }
        }
    }

    #endregion

    #endregion
}
