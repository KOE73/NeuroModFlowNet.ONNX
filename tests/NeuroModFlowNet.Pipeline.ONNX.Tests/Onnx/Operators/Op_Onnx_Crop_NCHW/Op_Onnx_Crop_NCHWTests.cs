using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Crop_NCHW;

public sealed class Op_Onnx_Crop_NCHWTests
{
    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task CropFP32NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP32NchwTensor(channels: 2, height: 4, width: 5);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Crop_FP32_NCHW(
            "input",
            "output",
            new Rect(1, 1, 3, 2),
            outputTransformKey: "cropToSource",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        var outputInfo = output.GetTensorTypeAndShape();
        Assert.Equal(TensorElementType.Float, outputInfo.ElementDataType);
        Assert.Equal([1, 2, 2, 3], outputInfo.Shape);

        Assert.Equal(
            [
                11, 12, 13,
                21, 22, 23,
                1011, 1012, 1013,
                1021, 1022, 1023
            ],
            output.GetTensorDataAsSpan<float>().ToArray());

        Assert.True(context.TryGet("cropToSource", out ICoordinateBackTransform _));
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task CropFP16NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP16NchwTensor(channels: 2, height: 4, width: 5);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Crop_FP16_NCHW(
            "input",
            "output",
            new Rect(1, 1, 3, 2),
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        var outputInfo = output.GetTensorTypeAndShape();
        Assert.Equal(TensorElementType.Float16, outputInfo.ElementDataType);
        Assert.Equal([1, 2, 2, 3], outputInfo.Shape);

        Half[] actual = MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>()).ToArray();
        Assert.Equal(
            [
                (Half)11, (Half)12, (Half)13,
                (Half)21, (Half)22, (Half)23,
                (Half)1011, (Half)1012, (Half)1013,
                (Half)1021, (Half)1022, (Half)1023
            ],
            actual);
    }

    static OrtValue CreateFP32NchwTensor(int channels, int height, int width)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [1, channels, height, width]);

        Span<float> data = tensor.GetTensorMutableDataAsSpan<float>();
        for(int channel = 0; channel < channels; channel++)
        {
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int index = channel * height * width + y * width + x;
                    data[index] = channel * 1000 + y * 10 + x;
                }
            }
        }

        return tensor;
    }

    static OrtValue CreateFP16NchwTensor(int channels, int height, int width)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float16,
            [1, channels, height, width]);

        Span<Half> data = MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorMutableDataAsSpan<Float16>());
        for(int channel = 0; channel < channels; channel++)
        {
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int index = channel * height * width + y * width + x;
                    data[index] = (Half)(channel * 1000 + y * 10 + x);
                }
            }
        }

        return tensor;
    }
}
