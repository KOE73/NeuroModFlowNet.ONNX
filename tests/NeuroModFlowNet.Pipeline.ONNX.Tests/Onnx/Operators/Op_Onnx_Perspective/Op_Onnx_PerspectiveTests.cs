using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Perspective;

public sealed class Op_Onnx_PerspectiveTests
{
    static readonly Point2f[] SourceQuad =
    [
        new(100, 50),
        new(599, 50),
        new(599, 549),
        new(100, 549)
    ];

    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PerspectiveU8NHWC_ReturnsExpectedPixels(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = CreateU8VisualSource(800, 650);
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_U8_NHWC(
            "input",
            "output",
            SourceQuad,
            new Size(500, 500),
            outputTransformKey: "perspectiveToSource",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.UInt8, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 500, 500, 3], output.GetTensorTypeAndShape().Shape);

        using Mat actual = OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);
        VisualAssert.PixelBgrNear(actual, new PixelExpectation(0, 0, source.At<Vec3b>(50, 100), "top-left"), tolerance: 2);
        VisualAssert.PixelBgrNear(actual, new PixelExpectation(499, 499, source.At<Vec3b>(549, 599), "bottom-right"), tolerance: 2);
        Assert.True(context.Get<ICoordinateBackTransform>("perspectiveToSource").TryMapBackward(new System.Numerics.Vector2(0, 0), out var mapped));
        Assert.Equal(100, mapped.X, precision: 3);
        Assert.Equal(50, mapped.Y, precision: 3);

        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_Perspective_U8_NHWC", "visual_quad_800x650_to_500x500");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PerspectiveFP32NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP32NchwTensor(3, 650, 800, CreateVisualGradientValue);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_FP32_NCHW(
            "input",
            "output",
            SourceQuad,
            new Size(500, 500),
            outputTransformKey: "perspectiveToSource",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 3, 500, 500], output.GetTensorTypeAndShape().Shape);
        Assert.True(context.TryGet("perspectiveToSource", out ICoordinateBackTransform _));

        SaveFP32Artifacts(input, output, executionBackend, "Op_Onnx_Perspective_FP32_NCHW", "visual_quad_800x650_to_500x500");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PerspectiveFP16NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP16NchwTensor(3, 650, 800, static (channel, y, x) => (Half)CreateVisualGradientValue(channel, y, x));
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_FP16_NCHW(
            "input",
            "output",
            SourceQuad,
            new Size(500, 500),
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 3, 500, 500], output.GetTensorTypeAndShape().Shape);

        SaveFP16Artifacts(input, output, executionBackend, "Op_Onnx_Perspective_FP16_NCHW", "visual_quad_800x650_to_500x500");
    }

    static Mat CreateU8VisualSource(int width, int height)
    {
        var image = new Mat(height, width, MatType.CV_8UC3);
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
                image.Set(y, x, new Vec3b((byte)(x & 0xFF), (byte)(y & 0xFF), (byte)((x + y) & 0xFF)));
        }

        return image;
    }

    static OrtValue CreateFP32NchwTensor(int channels, int height, int width, Func<int, int, int, float> valueFactory)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float, [1, channels, height, width]);
        Span<float> data = tensor.GetTensorMutableDataAsSpan<float>();

        for(int channel = 0; channel < channels; channel++)
        for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
            data[channel * height * width + y * width + x] = valueFactory(channel, y, x);

        return tensor;
    }

    static OrtValue CreateFP16NchwTensor(int channels, int height, int width, Func<int, int, int, Half> valueFactory)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, [1, channels, height, width]);
        Span<Half> data = MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorMutableDataAsSpan<Float16>());

        for(int channel = 0; channel < channels; channel++)
        for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
            data[channel * height * width + y * width + x] = valueFactory(channel, y, x);

        return tensor;
    }

    static float CreateVisualGradientValue(int channel, int y, int x) =>
        channel switch
        {
            0 => (x % 256) / 255f,
            1 => (y % 256) / 255f,
            2 => ((x + y) % 256) / 255f,
            _ => 0f
        };

    static void SaveFP32Artifacts(OrtValue input, OrtValue output, InferenceBackend executionBackend, string operationName, string caseName)
    {
        using Mat source = CreatePreviewFromFP32Nchw(input);
        using Mat actual = CreatePreviewFromFP32Nchw(output);
        SaveArtifacts(source, actual, executionBackend, operationName, caseName);
    }

    static void SaveFP16Artifacts(OrtValue input, OrtValue output, InferenceBackend executionBackend, string operationName, string caseName)
    {
        using Mat source = CreatePreviewFromFP16Nchw(input);
        using Mat actual = CreatePreviewFromFP16Nchw(output);
        SaveArtifacts(source, actual, executionBackend, operationName, caseName);
    }

    static void SaveArtifacts(Mat source, Mat actual, InferenceBackend executionBackend, string operationName, string caseName)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;
        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} source: {caseName}",
            source,
            environment.CreateOperationArtifactPath("Onnx", operationName, executionBackend.ToString(), caseName, "source"));
        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} actual: {caseName}",
            actual,
            environment.CreateOperationArtifactPath("Onnx", operationName, executionBackend.ToString(), caseName, "actual"));
    }

    static Mat CreatePreviewFromFP32Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(tensor.GetTensorDataAsSpan<float>(), checked((int)shape[1]), checked((int)shape[2]), checked((int)shape[3]), static value => NormalizePreviewByte(value));
    }

    static Mat CreatePreviewFromFP16Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorDataAsSpan<Float16>()), checked((int)shape[1]), checked((int)shape[2]), checked((int)shape[3]), static value => NormalizePreviewByte((float)value));
    }

    static Mat CreatePreviewFromNchw<T>(ReadOnlySpan<T> data, int channels, int height, int width, Func<T, byte> convert)
    {
        var image = new Mat(height, width, MatType.CV_8UC3);
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
                image.Set(y, x, new Vec3b(ReadChannel(data, channels, height, width, 0, y, x, convert), ReadChannel(data, channels, height, width, 1, y, x, convert), ReadChannel(data, channels, height, width, 2, y, x, convert)));
        }

        return image;
    }

    static byte ReadChannel<T>(ReadOnlySpan<T> data, int channels, int height, int width, int channel, int y, int x, Func<T, byte> convert)
    {
        int sourceChannel = Math.Min(channel, channels - 1);
        return convert(data[sourceChannel * height * width + y * width + x]);
    }

    static byte NormalizePreviewByte(float value)
    {
        float scaled = value is >= 0f and <= 1f ? value * 255f : value % 256f;
        if(scaled < 0)
            scaled += 256f;

        return (byte)Math.Clamp((int)MathF.Round(scaled), 0, 255);
    }
}
