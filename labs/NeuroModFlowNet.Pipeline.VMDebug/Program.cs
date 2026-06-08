using Spectre.Console;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Demo.Assets;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Entry point for the camera-based Pipeline VM debugging lab.
/// </summary>
public static class Program
{
    public static async Task Main(string[] args)
    {
        OnnxRuntimePathHelper.InitFromConfig();
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        LabSettings settings = LabSettings.FromConfig();

        // CLI args override the App.config mode if provided.
        LabMode mode = args.Length > 0 && Enum.TryParse<LabMode>(args[0], ignoreCase: true, out LabMode parsed)
            ? parsed
            : settings.Mode;

        using var stopTokenSource = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            stopTokenSource.Cancel();
        };

        try
        {
            string textObbModelPath = await AssetsManager.GetAssetPathAsync(
                ModelNaming.GetFileName("img-text-to-obb", precision: settings.ObbModelPrecision, isByteBgr: settings.ObbModelUseByteBgr)).ConfigureAwait(false);

            await RunModeAsync(mode, settings, textObbModelPath, stopTokenSource.Token).ConfigureAwait(false);
        }
        catch(Exception exception)
        {
            AnsiConsole.WriteException(exception, ExceptionFormats.ShortenEverything);
        }
    }

    static Task RunModeAsync(LabMode mode, LabSettings settings, string obbModelPath, CancellationToken cancellationToken)
    {
        return mode switch
        {
            LabMode.SingleStream => RunSingleStreamAsync(settings, obbModelPath, cancellationToken),
            LabMode.MultiStream  => RunMultiStreamAsync(settings, obbModelPath, cancellationToken),
            _                    => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown lab mode.")
        };
    }

    // ── Single-stream (original behaviour) ────────────────────────────────────

    static async Task RunSingleStreamAsync(LabSettings settings, string obbModelPath, CancellationToken cancellationToken)
    {
        // Single stream runs the OBB instruction directly inside the VM program.
        VmProgram program = VmProgramFactory.CreateProgram(obbModelPath, includeObb: true);
        var host = new CameraVmHost(program, CameraVmHostOptions.FromSettings(settings));
        var dashboard = new SpectreVmDashboard();

        await dashboard.RunAsync(host, cancellationToken).ConfigureAwait(false);
    }

    // ── Multi-stream (4 emulated VM workers + OBB batch service) ──────────────

    static async Task RunMultiStreamAsync(LabSettings settings, string obbModelPath, CancellationToken cancellationToken)
    {
        // VM program: crop + resize only. OBB model lives in ObbBatchService (batched).
        VmProgram program = VmProgramFactory.CreateProgram(obbModelPath, includeObb: false);

        var host = new MultiStreamHost(program, MultiStreamOptions.FromSettings(settings), obbModelPath);
        var dashboard = new MultiStreamDashboard();

        await dashboard.RunAsync(host, cancellationToken).ConfigureAwait(false);
    }
}
