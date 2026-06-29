using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_PadResize;

public sealed class OpOnnxPadResizePatternCase : IDisposable
{
    public required string Name { get; init; }

    public required Mat Source { get; init; }

    public required Size TargetSize { get; init; }

    public required PadResizeMode Mode { get; init; }

    public int Stride { get; init; } = 32;

    public byte PadValue { get; init; } = 114;

    public required Size ExpectedSize { get; init; }

    public required IReadOnlyList<PixelExpectation> ExpectedPixels { get; init; }

    public int Tolerance { get; init; }

    public void Dispose() => Source.Dispose();
}
