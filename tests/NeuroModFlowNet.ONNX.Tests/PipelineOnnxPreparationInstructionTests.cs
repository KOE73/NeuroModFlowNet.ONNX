using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX;
using System.Runtime.InteropServices;

namespace NeuroModFlowNet.ONNX.Tests;

public sealed class PipelineOnnxPreparationInstructionTests
{
    [Fact]
    public async Task BgrU8HwcToRgbFP32NchwDiv255_ConvertsLayoutChannelsAndRange()
    {
        using OrtValue input = CreateBgrU8HwcInput(
            width: 2,
            height: 1,
            [
                10, 20, 30,
                40, 50, 60
            ]);

        await using var context = CreateContext(input);
        using var instruction = new Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
            "input",
            "output",
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        var outputInfo = output.GetTensorTypeAndShape();

        Assert.Equal(TensorElementType.Float, outputInfo.ElementDataType);
        Assert.Equal([1, 3, 1, 2], outputInfo.Shape);

        float[] actual = output.GetTensorDataAsSpan<float>().ToArray();
        float[] expected =
        [
            30f / 255f, 60f / 255f,
            20f / 255f, 50f / 255f,
            10f / 255f, 40f / 255f
        ];

        AssertEqualWithinTolerance(expected, actual, tolerance: 0.000001f);
    }

    [Fact]
    public async Task BgrU8HwcToRgbFP16NchwDiv255_ConvertsLayoutChannelsRangeAndPrecision()
    {
        using OrtValue input = CreateBgrU8HwcInput(
            width: 2,
            height: 1,
            [
                10, 20, 30,
                40, 50, 60
            ]);

        await using var context = CreateContext(input);
        using var instruction = new Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255(
            "input",
            "output",
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        var outputInfo = output.GetTensorTypeAndShape();

        Assert.Equal(TensorElementType.Float16, outputInfo.ElementDataType);
        Assert.Equal([1, 3, 1, 2], outputInfo.Shape);

        Half[] actual = MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>()).ToArray();
        Half[] expected =
        [
            (Half)(30f / 255f), (Half)(60f / 255f),
            (Half)(20f / 255f), (Half)(50f / 255f),
            (Half)(10f / 255f), (Half)(40f / 255f)
        ];

        Assert.Equal(expected, actual);
    }

    static OrtValue CreateBgrU8HwcInput(int width, int height, ReadOnlySpan<byte> bgrData)
    {
        long[] shape = [1, height, width, 3];
        int expectedLength = checked(width * height * 3);

        if(bgrData.Length != expectedLength)
            throw new ArgumentException($"Expected {expectedLength} BGR bytes, actual: {bgrData.Length}.", nameof(bgrData));

        OrtValue input = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.UInt8, shape);
        bgrData.CopyTo(input.GetTensorMutableDataAsSpan<byte>());
        return input;
    }

    static VmRunContext CreateContext(OrtValue input)
    {
        var context = new VmRunContext(
            new VmRunIdentity("test", 0, DateTimeOffset.UtcNow),
            new VmGlobalMemory(),
            new VmSyncGateRegistry());

        context.Set("input", input);
        return context;
    }

    static void AssertEqualWithinTolerance(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        for(int index = 0; index < expected.Length; index++)
            Assert.True(Math.Abs(expected[index] - actual[index]) <= tolerance, $"Index {index}: expected {expected[index]}, actual {actual[index]}.");
    }
}
