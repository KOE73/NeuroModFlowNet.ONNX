using NeuroModFlowNet.Pipeline.Video.Nvdec;
using System.Configuration;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.VideoOrt;

/// <summary>
/// EN: Lab for the video → ONNX Runtime chain only: FFmpeg 9 demux, NVDEC decode into CUDA memory, zero-copy
/// <c>OrtValue</c> over the surface, NV12→BGR as an ONNX op on the GPU, then the usual VM ops. <c>verify</c> checks
/// pixels against CPU decoding, <c>bench</c> measures N parallel streams.
///
/// RU: Лаборатория только для цепочки видео → ONNX Runtime: демультиплексор FFmpeg 9, декод NVDEC в память CUDA,
/// zero-copy <c>OrtValue</c> поверх поверхности, NV12→BGR ONNX-операцией на GPU, дальше обычные VM-операции.
/// <c>verify</c> сверяет пиксели с CPU-декодом, <c>bench</c> измеряет N параллельных потоков.
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        try
        {
            VideoOrtOptions options = VideoOrtOptions.Parse(args);

            OnnxRuntimePathHelper.InitFromConfig();
            NeuroModFlowNet.Pipeline.ONNX.RuntimeOnnxOperatorKernelCache.SessionSharing = options.SessionSharing;
            string librariesPath = ConfigurationManager.AppSettings["FFmpegLibrariesPath"]
                ?? throw new InvalidOperationException("FFmpegLibrariesPath is not configured.");
            FfmpegRuntime.Initialize(librariesPath);
            Console.WriteLine($"FFmpeg {FfmpegRuntime.Version} from {librariesPath}");

            using var stopSource = new CancellationTokenSource();
            Console.CancelKeyPress += (_, eventArgs) =>
            {
                eventArgs.Cancel = true;
                stopSource.Cancel();
            };

            return options.Command == "verify"
                ? await VerifyCommand.RunAsync(options).ConfigureAwait(false)
                : await BenchCommand.RunAsync(options, stopSource.Token).ConfigureAwait(false);
        }
        catch(ArgumentException exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 64;
        }
        catch(Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }
}
