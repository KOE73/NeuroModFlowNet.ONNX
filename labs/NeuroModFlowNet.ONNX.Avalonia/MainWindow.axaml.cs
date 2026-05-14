using Avalonia.Controls;
using Avalonia.Threading;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;
using NeuroModFlowNet.ONNX.Demo.Assets;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia;

/// <summary>
/// EN: Represents the main window that hosts the realtime OCR lab workspace.
/// RU: Представляет главное окно, в котором размещается realtime OCR lab workspace.
/// </summary>
/// <remarks>
/// EN: Connects the settings panel, video scene, metrics panel, and background inference engine. The window receives
/// frame data from the engine and applies UI updates on the Avalonia UI thread.
/// RU: Связывает панель настроек, область видео, панель метрик и background inference engine. Окно получает данные
/// кадров от engine и обновляет UI в Avalonia UI thread.
/// </remarks>
public partial class MainWindow : global::Avalonia.Controls.Window
{
    readonly RecognitionOptions recognitionOptions = new();
    readonly RealTimeInferenceEngine engine;

    public MainWindow()
    {
        InitializeComponent();

        engine = new RealTimeInferenceEngine(RealTimeAvaloniaSettings.FromConfig(), recognitionOptions);
        engine.FrameReady += OnFrameReady;
        engine.StatusChanged += OnStatusChanged;
        engine.ModelInfoChanged += OnModelInfoChanged;

        ModeNavigationView_ModeNavigation.BindOptions(recognitionOptions);
        ModeNavigationView_ModeNavigation.SetVideoSourceInfo(VideoCaptureConfig.GetSourceInfo());
        OcrSettingsView_OcrSettings.BindOptions(recognitionOptions);
        ModeNavigationView_ModeNavigation.SetStatus("Ready");
        ModeNavigationView_ModeNavigation.StartRequested += async (_, _) => await StartEngineAsync();
        ModeNavigationView_ModeNavigation.StopRequested += async (_, _) => await engine.StopAsync();
        ModeNavigationView_ModeNavigation.PauseRequested += (_, _) => engine.Pause();
        ModeNavigationView_ModeNavigation.PlayRequested += (_, _) => engine.Play();
        ModeNavigationView_ModeNavigation.StepRequested += (_, _) => engine.StepFrame();
        OcrSettingsView_OcrSettings.OptionsChanged += (_, _) => OcrSettingsView_OcrSettings.Refresh();
        ModeNavigationView_ModeNavigation.OptionsChanged += (_, _) => ModeNavigationView_ModeNavigation.Refresh();

        ShowInitialFrame();
        Dispatcher.UIThread.Post(async () => await StartEngineAsync());
    }

    async Task StartEngineAsync()
    {
        ModeNavigationView_ModeNavigation.SetStatus("Starting...");
        await engine.StartAsync();
    }

    void OnFrameReady(object? sender, RealTimeOneFrameData update)
    {
        Dispatcher.UIThread.Post(() =>
        {
            using(update)
            {
                VideoSceneView_LiveVideoScene.UpdateFrame(update.Frame, update.DetFrame, update.Overlay);
                VideoSceneView_LiveVideoScene.UpdateRecognition(update.RecognitionItems);
                MetricsPanelView_RuntimeMetrics.UpdateMetrics(update.Metrics);
            }
        });
    }

    void OnStatusChanged(object? sender, string status)
    {
        Dispatcher.UIThread.Post(() => ModeNavigationView_ModeNavigation.SetStatus(status));
    }

    void OnModelInfoChanged(object? sender, IReadOnlyList<RuntimeModelInfo> modelInfos)
    {
        Dispatcher.UIThread.Post(() => ModeNavigationView_ModeNavigation.UpdateModelInfo(modelInfos));
    }

    void ShowInitialFrame()
    {
        using var placeholder = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(
            placeholder,
            "Press Start",
            new Point(470, 360),
            HersheyFonts.HersheySimplex,
            1.5,
            new Scalar(80, 80, 80),
            2,
            LineTypes.AntiAlias);

        using var bgra = new Mat();
        Cv2.CvtColor(placeholder, bgra, ColorConversionCodes.BGR2BGRA);
        using var detPlaceholder = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(
            detPlaceholder,
            "PaddleOCR Det",
            new Point(420, 360),
            HersheyFonts.HersheySimplex,
            1.2,
            new Scalar(80, 80, 80),
            2,
            LineTypes.AntiAlias);
        using var detBgra = new Mat();
        Cv2.CvtColor(detPlaceholder, detBgra, ColorConversionCodes.BGR2BGRA);
        VideoSceneView_LiveVideoScene.UpdateFrame(bgra, detBgra, new FrameOverlaySnapshot(bgra.Width, bgra.Height, [], []));
    }

    protected override void OnClosed(EventArgs e)
    {
        engine.Dispose();
        recognitionOptions.Dispose();
        base.OnClosed(e);
    }
}
