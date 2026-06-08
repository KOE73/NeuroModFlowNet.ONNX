using NeuroModFlowNet.ONNX.Graph.Builders;

namespace NeuroModFlowNet.ONNX.Tests;

public sealed class OnnxRuntimeContextModelSourceTests
{
    [Fact]
    public void ByteModelSource_ShouldRunGeneratedCropModel()
    {
        byte[] modelBytes = CropBuilder.Build(
            sourceWidth: 3,
            sourceHeight: 2,
            x1: 1,
            y1: 0,
            x2: 3,
            y2: 2,
            channels: 1);

        using var context = new OnnxRuntimeContext(
            modelBytes,
            InferenceBackend.Cpu,
            displayName: "test-crop.onnx");

        Assert.Equal(OnnxRuntimeModelSourceKind.Bytes, context.Model.ModelSource.Kind);
        Assert.Equal("test-crop.onnx", context.Model.ModelSource.DisplayName);

        context.InitInputPersistentValue(CropBuilder.InputName, [1, 2, 3, 1]);
        context.InitOutputPersistentValue(CropBuilder.OutputName, [1, 2, 2, 1]);

        Span<byte> input = context.GetInputBuffer<byte>(CropBuilder.InputName);
        byte[] sourcePixels =
        [
            1, 2, 3,
            4, 5, 6
        ];
        sourcePixels.CopyTo(input);

        context.Run();

        byte[] actual = context.GetTensorDataAsSpan<byte>(CropBuilder.OutputName).ToArray();
        byte[] expected =
        [
            2, 3,
            5, 6
        ];

        Assert.Equal(expected, actual);
    }
}
