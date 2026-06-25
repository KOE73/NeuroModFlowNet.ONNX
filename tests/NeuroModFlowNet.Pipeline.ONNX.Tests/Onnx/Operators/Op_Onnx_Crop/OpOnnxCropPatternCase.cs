using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Crop;

public sealed class OpOnnxCropPatternCase : IDisposable
{
    public required string Name { get; init; }

    public required Mat Source { get; init; }

    public required Rect CropRect { get; init; }

    public required Size ExpectedSize { get; init; }

    public required IReadOnlyList<PixelExpectation> ExpectedPixels { get; init; }

    public int Tolerance { get; init; }

    public int? Seed { get; init; }

    public void Dispose() => Source.Dispose();
}
