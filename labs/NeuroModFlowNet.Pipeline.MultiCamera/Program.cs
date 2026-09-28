using NeuroModFlowNet.ONNX;
using Spectre.Console;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Entry point of the multi-camera Pipeline VM lab: N cameras (looped video folders and/or RTSP), one
/// <c>VmController</c> per camera with the program chain <c>[prepare:&lt;camera&gt;, common]</c>, shared batched
/// model endpoints. Usage: <c>NeuroModFlowNet.Pipeline.MultiCamera [cameras.json] [--no-preview] [--plain] [--profile] [--decoder cpu|nvdec] [--snapshots &lt;dir&gt;]</c>.
///
/// RU: Точка входа лаборатории многокамерного VM-пайплайна: N камер (зацикленные папки с видео и/или RTSP), один
/// <c>VmController</c> на камеру с цепочкой <c>[prepare:&lt;camera&gt;, common]</c>, общие батчевые endpoint-ы моделей.
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        OnnxRuntimePathHelper.InitFromConfig();

        // Every camera controller runs the same runtime kernels concurrently: shared sessions would serialize them.
        NeuroModFlowNet.Pipeline.ONNX.RuntimeOnnxOperatorKernelCache.SessionSharing = NeuroModFlowNet.Pipeline.ONNX.RuntimeKernelSessionSharing.PerCaller;
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        bool showPreview = !args.Contains("--no-preview", StringComparer.OrdinalIgnoreCase);
        bool plainOutput = Console.IsOutputRedirected || args.Contains("--plain", StringComparer.OrdinalIgnoreCase);
        bool profile = args.Contains("--profile", StringComparer.OrdinalIgnoreCase);
        string? decoderOverride = ReadOption(args, "--decoder");
        string? snapshotDirectory = ReadOption(args, "--snapshots");
        string[] positional = args
            .Where(static argument => !argument.StartsWith("--", StringComparison.Ordinal))
            .Where(argument => !string.Equals(argument, snapshotDirectory, StringComparison.Ordinal))
            .Where(argument => !string.Equals(argument, decoderOverride, StringComparison.Ordinal))
            .ToArray();

        using var stopTokenSource = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            stopTokenSource.Cancel();
        };

        try
        {
            string configPath = MultiCameraConfigLoader.ResolveConfigPath(positional);
            MultiCameraConfig config = MultiCameraConfigLoader.Load(configPath);
            if(decoderOverride is not null)
            {
                // --decoder cpu|nvdec switches every camera without editing the config.
                config = config with
                {
                    Cameras = config.Cameras
                        .Select(camera => camera with { Source = camera.Source with { Decoder = decoderOverride, HwAcceleration = "none" } })
                        .ToArray()
                };

                foreach(CameraConfig camera in config.Cameras)
                    camera.Validate();
            }
            if(config.EnabledCameras.Any(static camera => camera.Source.IsNvdec))
            {
                string librariesPath = System.Configuration.ConfigurationManager.AppSettings["FFmpegLibrariesPath"]
                    ?? throw new InvalidOperationException("FFmpegLibrariesPath is required for decoder 'nvdec'.");
                NeuroModFlowNet.Pipeline.Video.Nvdec.FfmpegRuntime.Initialize(librariesPath);
            }
            PrintSummary(configPath, config);

            await using MultiCameraHost host = await MultiCameraHost.CreateAsync(config).ConfigureAwait(false);
            PrintPrograms(host);

            if(snapshotDirectory is not null)
            {
                foreach(CameraWorker worker in host.Workers)
                    worker.SnapshotDirectory = snapshotDirectory;
            }

            var dashboard = new MultiCameraDashboard(host, showPreview && config.Preview.Enabled, plainOutput, profile);
            await dashboard.RunAsync(stopTokenSource).ConfigureAwait(false);
            if(profile && !plainOutput)
                dashboard.PrintProfile();

            return 0;
        }
        catch(OperationCanceledException) when(stopTokenSource.IsCancellationRequested)
        {
            return 0;
        }
        catch(Exception exception)
        {
            AnsiConsole.WriteException(exception, ExceptionFormats.ShortenEverything);
            return 1;
        }
    }

    static string? ReadOption(string[] args, string name)
    {
        int index = Array.FindIndex(args, argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    static void PrintSummary(string configPath, MultiCameraConfig config)
    {
        AnsiConsole.MarkupLine($"[grey]config:[/] {Markup.Escape(configPath)}");
        AnsiConsole.MarkupLine($"[grey]backend:[/] {config.Backend}, OCR {(config.Ocr.Enabled ? "on" : "off")}, preview {(config.Preview.Enabled ? "on" : "off")}");
        AnsiConsole.MarkupLine($"[grey]detector:[/] {config.Detector.Kind}, {config.Detector.Backend ?? config.Backend}, {config.Detector.InputMode}");

        foreach(CameraConfig camera in config.EnabledCameras)
        {
            string transforms = camera.Transforms.Count == 0
                ? "none"
                : string.Join(" -> ", camera.Transforms.Select(static transform => transform.Type));
            AnsiConsole.MarkupLine($"[grey]camera[/] [bold]{Markup.Escape(camera.Id)}[/]: {camera.Resolution.Width}x{camera.Resolution.Height}, {Markup.Escape(camera.Source.Kind)} / {Markup.Escape(camera.Source.Decoder)}, transforms: {Markup.Escape(transforms)}, detector {camera.Detector.InputWidth}x{camera.Detector.InputHeight} {Markup.Escape(Path.GetFileName(camera.Detector.ModelPath))}");
        }
    }

    static void PrintPrograms(MultiCameraHost host)
    {
        foreach(CameraWorker worker in host.Workers)
        {
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(worker.Id)}[/] source: {Markup.Escape(worker.SourceDescription)}, rect {worker.Geometry.Width}x{worker.Geometry.Height}");
            foreach(VmProgram program in worker.Programs)
            {
                string instructions = string.Join(", ", program.Instructions.Select(static instruction => instruction.Descriptor.Name));
                AnsiConsole.MarkupLine($"  [grey]{Markup.Escape(program.Name ?? "program")}[/]: {Markup.Escape(instructions)}");
            }
        }
    }
}
