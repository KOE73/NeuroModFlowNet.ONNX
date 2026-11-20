using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: A completed CUDA BGR U8 NHWC tensor and its draw commands. The frame owns the tensor.
///
/// RU: Готовый тензор CUDA BGR U8 NHWC и команды рисования. Кадр владеет тензором.
/// </summary>
public sealed class VideoInputFrame(OrtValue ownedTensor, long sequence, ReadOnlyMemory<Direct2DRectangle> rectangles) : IDisposable
{
    OrtValue? tensor = ownedTensor ?? throw new ArgumentNullException(nameof(ownedTensor));

    internal OrtValue Tensor => tensor ?? throw new ObjectDisposedException(nameof(VideoInputFrame));
    internal long SubmittedAt { get; } = Stopwatch.GetTimestamp();
    public long Sequence { get; } = sequence;
    public ReadOnlyMemory<Direct2DRectangle> Rectangles { get; } = rectangles;

    public void Dispose() => Interlocked.Exchange(ref tensor, null)?.Dispose();
}
