using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Rotate90;

public sealed class OpOnnxRotate90PatternCase : IDisposable
{
    public required string Name { get; init; }

    public required Mat Source { get; init; }

    public required Rotate90Mode Mode { get; init; }

    public required Size ExpectedSize { get; init; }

    public required IReadOnlyList<PixelExpectation> ExpectedPixels { get; init; }

    public int Tolerance { get; init; }

    public void Dispose() => Source.Dispose();
}
