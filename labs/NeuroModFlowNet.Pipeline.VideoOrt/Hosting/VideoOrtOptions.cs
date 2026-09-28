using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.VideoOrt;

/// <summary>Command line of the video → ORT lab.</summary>
internal sealed record VideoOrtOptions(
    string Command,
    IReadOnlyList<string> Sources,
    InferenceBackend Backend,
    bool UseFence,
    Nv12ColorMatrix Matrix,
    int Frames,
    double MaxMeanDifference,
    string OutputDirectory,
    int Streams,
    int Seconds,
    int MaxInFlight,
    bool DecodeOnly,
    int ResizeWidth,
    int ResizeHeight,
    NeuroModFlowNet.Pipeline.ONNX.RuntimeKernelSessionSharing SessionSharing)
{
    public const string Usage = """
        usage:
          verify <file> [--frames 200] [--backend TensorRt|Cuda] [--matrix Bt601Full] [--no-fence] [--out dir] [--threshold 3]
          bench <file|rtsp> [more sources...] [--streams 10] [--seconds 30] [--backend TensorRt] [--inflight 3]
                [--decode-only] [--no-fence] [--resize 1280x720] [--sessions PerCaller|Shared]
        """;

    public static VideoOrtOptions Parse(string[] args)
    {
        if(args.Length < 2 || args[0] is not ("verify" or "bench"))
            throw new ArgumentException(Usage);

        var sources = new List<string>();
        var named = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        for(int index = 1; index < args.Length; index++)
        {
            if(!args[index].StartsWith("--", StringComparison.Ordinal))
            {
                sources.Add(args[index]);
                continue;
            }

            bool isFlag = args[index] is "--no-fence" or "--decode-only";
            named[args[index]] = isFlag ? null : args[++index];
        }

        if(sources.Count == 0)
            throw new ArgumentException("At least one source is required.\n" + Usage);

        string? resize = named.GetValueOrDefault("--resize") ?? "1280x720";
        string[] resizeParts = resize.Split('x');

        return new VideoOrtOptions(
            Command: args[0],
            Sources: sources,
            Backend: Enum.Parse<InferenceBackend>(named.GetValueOrDefault("--backend") ?? "TensorRt", ignoreCase: true),
            UseFence: !named.ContainsKey("--no-fence"),
            Matrix: Enum.Parse<Nv12ColorMatrix>(named.GetValueOrDefault("--matrix") ?? "Bt601Full", ignoreCase: true),
            Frames: int.Parse(named.GetValueOrDefault("--frames") ?? "200"),
            MaxMeanDifference: double.Parse(named.GetValueOrDefault("--threshold") ?? "3", System.Globalization.CultureInfo.InvariantCulture),
            OutputDirectory: named.GetValueOrDefault("--out") ?? Path.Combine(Path.GetTempPath(), "NeuroModFlowNet.VideoOrt"),
            Streams: int.Parse(named.GetValueOrDefault("--streams") ?? sources.Count.ToString()),
            Seconds: int.Parse(named.GetValueOrDefault("--seconds") ?? "30"),
            MaxInFlight: int.Parse(named.GetValueOrDefault("--inflight") ?? "3"),
            DecodeOnly: named.ContainsKey("--decode-only"),
            ResizeWidth: int.Parse(resizeParts[0]),
            ResizeHeight: int.Parse(resizeParts[1]),
            SessionSharing: Enum.Parse<NeuroModFlowNet.Pipeline.ONNX.RuntimeKernelSessionSharing>(named.GetValueOrDefault("--sessions") ?? "PerCaller", ignoreCase: true));
    }
}
