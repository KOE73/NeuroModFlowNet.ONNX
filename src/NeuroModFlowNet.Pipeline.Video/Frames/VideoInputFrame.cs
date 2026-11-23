using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: A completed CUDA BGR U8 NHWC tensor and the named input values for the render VM.
///
/// RU: Готовый тензор CUDA BGR U8 NHWC и именованные входные значения для render VM.
/// </summary>
/// <remarks>
/// EN: The frame owns the tensor. Additional register values are borrowed and must remain valid and unchanged while the frame is queued or rendered.
///
/// RU: Кадр владеет тензором. Дополнительные значения регистров передаются без владения и должны оставаться доступными и неизменными в очереди и при рисовании.
/// </remarks>
public sealed class VideoInputFrame : IDisposable
{
    internal const string FrameRegister = "frame";
    internal const string SurfaceRegister = "surface";
    internal const string HandoffRegister = "video.handoff";

    OrtValue? tensor;

    public VideoInputFrame(OrtValue ownedTensor, long sequence, ReadOnlyMemory<Direct2DRectangle> rectangles)
    {
        tensor = ownedTensor ?? throw new ArgumentNullException(nameof(ownedTensor));
        Sequence = sequence;
        Rectangles = rectangles;
        Registers = new Dictionary<string, object?>(StringComparer.Ordinal) { [RectangleVideoDrawer.RegisterKey] = rectangles };
    }

    public VideoInputFrame(OrtValue ownedTensor, long sequence, IReadOnlyDictionary<string, object?> registers)
    {
        ArgumentNullException.ThrowIfNull(ownedTensor);
        ArgumentNullException.ThrowIfNull(registers);
        var snapshot = new Dictionary<string, object?>(registers.Count, StringComparer.Ordinal);
        foreach(var (key, value) in registers)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(key);
            if(key is FrameRegister or SurfaceRegister or HandoffRegister)
                throw new ArgumentException($"Register '{key}' is managed by the video render VM.", nameof(registers));
            snapshot.Add(key, value);
        }

        Registers = snapshot;
        Sequence = sequence;
        tensor = ownedTensor;
    }

    internal OrtValue Tensor => tensor ?? throw new ObjectDisposedException(nameof(VideoInputFrame));
    internal IReadOnlyDictionary<string, object?> Registers { get; }
    internal long SubmittedAt { get; } = Stopwatch.GetTimestamp();
    public long Sequence { get; }
    public ReadOnlyMemory<Direct2DRectangle> Rectangles { get; }

    public void Dispose() => Interlocked.Exchange(ref tensor, null)?.Dispose();
}
