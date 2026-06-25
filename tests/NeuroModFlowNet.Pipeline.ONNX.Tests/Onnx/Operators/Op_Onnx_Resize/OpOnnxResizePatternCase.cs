using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Resize;

public sealed class OpOnnxResizePatternCase : IDisposable
{
    public required string Name { get; init; }

    public required Mat Source { get; init; }

    public required Size TargetSize { get; init; }

    public required IReadOnlyList<PixelExpectation> ExpectedPixels { get; init; }

    public int Tolerance { get; init; }

    public void Dispose() => Source.Dispose();
}
