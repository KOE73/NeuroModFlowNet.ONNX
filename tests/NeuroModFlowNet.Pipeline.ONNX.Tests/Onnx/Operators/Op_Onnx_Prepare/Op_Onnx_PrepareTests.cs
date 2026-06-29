using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Prepare;

public sealed class Op_Onnx_PrepareTests
{
    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task BgrU8HwcToRgbFP32NchwDiv255_ConvertsOnBackendMatrix(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateBgrU8HwcInput(width: 2, height: 1, [10, 20, 30, 40, 50, 60]);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
            "input",
            "output",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 3, 1, 2], output.GetTensorTypeAndShape().Shape);
        AssertEqualWithinTolerance(
            [30f / 255f, 60f / 255f, 20f / 255f, 50f / 255f, 10f / 255f, 40f / 255f],
            output.GetTensorDataAsSpan<float>().ToArray(),
            tolerance: 0.000001f);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task BgrU8HwcToRgbFP16NchwDiv255_ConvertsOnBackendMatrix(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateBgrU8HwcInput(width: 2, height: 1, [10, 20, 30, 40, 50, 60]);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255(
            "input",
            "output",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 3, 1, 2], output.GetTensorTypeAndShape().Shape);
        Assert.Equal(
            [(Half)(30f / 255f), (Half)(60f / 255f), (Half)(20f / 255f), (Half)(50f / 255f), (Half)(10f / 255f), (Half)(40f / 255f)],
            MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>()).ToArray());
    }

    static OrtValue CreateBgrU8HwcInput(int width, int height, ReadOnlySpan<byte> bgrData)
    {
        OrtValue input = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.UInt8, [1, height, width, 3]);
        bgrData.CopyTo(input.GetTensorMutableDataAsSpan<byte>());
        return input;
    }

    static void AssertEqualWithinTolerance(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        for(int index = 0; index < expected.Length; index++)
            Assert.True(Math.Abs(expected[index] - actual[index]) <= tolerance, $"Index {index}: expected {expected[index]}, actual {actual[index]}.");
    }
}
