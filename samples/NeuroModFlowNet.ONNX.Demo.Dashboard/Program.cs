using System.Diagnostics;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;
using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal class Program
{
    private const InferenceBackend Backend = InferenceBackend.TensorRt;
    private const string Precision = "fp32";
    private const int InputSize = 640;
    private const bool IsByteBgr = false;

    private static readonly HashSet<ModelSlot> EnabledSlots = new()
    {
        ModelSlot.Raw,
        ModelSlot.Box,
        ModelSlot.Obb,
        ModelSlot.Seg,
        ModelSlot.Pose,
        ModelSlot.Cls,
    };

    static async Task Main(string[] args)
    {
        OnnxRuntimePathHelper.InitFromConfig();

        List<IDashboardFrameProcessor> processors = [];

        try
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new FigletText("NMFN DASHBOARD").Color(Color.Cyan1));
            AnsiConsole.MarkupLine("[bold white]Initializing Multi-Model Inference System...[/]");
            AnsiConsole.MarkupLine($"[grey]Slots: {string.Join(", ", EnabledSlots)}[/]");

            var settings = new DashboardModelSettings(Backend, InputSize, Precision, IsByteBgr);
            processors = CreateProcessors(settings);
            IDashboardFrameProcessor[] activeProcessors = processors
                .Where(processor => EnabledSlots.Contains(processor.Slot))
                .ToArray();

            foreach(IDashboardFrameProcessor processor in activeProcessors)
            {
                if(processor.Slot != ModelSlot.Raw)
                    AnsiConsole.MarkupLine($"[cyan]Loading {processor.Title} model...[/]");

                await processor.InitializeAsync();
            }

            using var capture = VideoCaptureConfig.CreateFromConfig();
            if(!capture.IsOpened())
                AnsiConsole.MarkupLine("[yellow]Warning: Camera not found — simulation (black frame) mode.[/]");

            using var painter = new DashboardPainter(1920, 1080);
            using var window = new Window("AI DASHBOARD [NeuroModFlowNet.ONNX]");

            var frameStopwatch = Stopwatch.StartNew();
            int frameCount = 0;
            double fps = 0;
            double inferenceTimeMs = 0;

            using var matFrame = new Mat();

            AnsiConsole.MarkupLine("[green]System Online. Press ESC or Q to exit.[/]\n");

            while(true)
            {
                CaptureFrame(capture, matFrame);

                using Mat letterboxed = matFrame.Letterbox(settings.InputSize, settings.InputSize, out LetterboxInfo letterboxInfo);
                var frameInfo = new DashboardFrameInfo(
                    letterboxInfo,
                    letterboxInfo.SourceWidth,
                    letterboxInfo.SourceHeight);

                var inferenceStopwatch = Stopwatch.StartNew();
                Task[] tasks = activeProcessors
                    .Where(processor => processor.IsEnabled)
                    .Select(processor => Task.Run(() => processor.Process(letterboxed, frameInfo)))
                    .ToArray();

                Task.WaitAll(tasks);
                inferenceTimeMs = inferenceStopwatch.Elapsed.TotalMilliseconds;

                List<DashboardView> views = processors
                    .Select(processor => new DashboardView(processor.Title, target => processor.Draw(target, frameInfo)))
                    .ToList();

                Mat canvas = painter.Draw(matFrame, views, fps, inferenceTimeMs);
                window.ShowImage(canvas);

                frameCount++;
                if(frameStopwatch.ElapsedMilliseconds > 1000)
                {
                    fps = frameCount * 1000.0 / frameStopwatch.ElapsedMilliseconds;
                    frameCount = 0;
                    frameStopwatch.Restart();
                }

                int key = Cv2.WaitKey(1);
                if(key == 27 || key == 'q' || key == 'Q') break;
            }
        }
        catch(Exception ex)
        {
            AnsiConsole.WriteException(ex, ExceptionFormats.ShortenEverything);
            AnsiConsole.MarkupLine("\n[bold red]Critical error.[/] Press any key to exit...");
            Console.ReadKey(true);
        }
        finally
        {
            foreach(IDashboardFrameProcessor processor in processors)
                processor.Dispose();
        }
    }

    private static List<IDashboardFrameProcessor> CreateProcessors(DashboardModelSettings settings) =>
    [
        new RawDashboardProcessor(),
        new YoloBoxDashboardProcessor(settings),
        new YoloObbDashboardProcessor(settings),
        new YoloSegDashboardProcessor(settings),
        new YoloPoseDashboardProcessor(settings),
        new YoloClsDashboardProcessor(settings),
    ];

    private static void CaptureFrame(VideoCapture capture, Mat matFrame)
    {
        capture.Read(matFrame);
        if(!matFrame.Empty()) return;

        using var dummy = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(
            dummy,
            "NO SIGNAL",
            new Point(480, 360),
            HersheyFonts.HersheySimplex,
            1.5,
            new Scalar(60, 60, 60),
            2);
        dummy.CopyTo(matFrame);
    }
}
