using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_NCHW_Geometry;

public sealed class Op_Onnx_NCHW_GeometryTests
{
    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task ResizeFP32NCHW_PreservesConstantTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP32NchwTensor(channels: 2, height: 4, width: 5, valueFactory: static (channel, _, _) => channel + 0.25f);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize_FP32_NCHW(
            "input",
            "output",
            new Size(8, 6),
            outputTransformKey: "resizeToSource",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 2, 6, 8], output.GetTensorTypeAndShape().Shape);
        AssertChannelConstant(output.GetTensorDataAsSpan<float>(), channels: 2, height: 6, width: 8, [0.25f, 1.25f]);
        Assert.True(context.TryGet("resizeToSource", out ICoordinateBackTransform _));
        await SaveResizeFP32VisualArtifacts(executionBackend);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task ResizeFP16NCHW_PreservesConstantTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP16NchwTensor(channels: 2, height: 4, width: 5, valueFactory: static (channel, _, _) => (Half)(channel + 0.25f));
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize_FP16_NCHW(
            "input",
            "output",
            new Size(8, 6),
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 2, 6, 8], output.GetTensorTypeAndShape().Shape);
        AssertChannelConstant(MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>()), channels: 2, height: 6, width: 8, [(Half)0.25f, (Half)1.25f]);
        await SaveResizeFP16VisualArtifacts(executionBackend);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PadResizeFP32NCHW_AddsTypedPadding(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP32NchwTensor(channels: 1, height: 2, width: 4, valueFactory: static (_, _, _) => 0.25f);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_FP32_NCHW(
            "input",
            "output",
            new Size(6, 6),
            outputTransformKey: "padResizeToSource",
            padValue: 0.5f,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal([1, 1, 6, 6], output.GetTensorTypeAndShape().Shape);
        ReadOnlySpan<float> data = output.GetTensorDataAsSpan<float>();
        Assert.Equal(0.5f, data[0]);
        Assert.Equal(0.25f, data[1 * 6 + 0]);
        Assert.Equal(0.25f, data[3 * 6 + 5]);
        Assert.Equal(0.5f, data[5 * 6 + 5]);
        Assert.True(context.TryGet("padResizeToSource", out ICoordinateBackTransform _));
        await SavePadResizeFP32VisualArtifacts(executionBackend);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PadResizeFP16NCHW_AddsTypedPadding(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP16NchwTensor(channels: 1, height: 2, width: 4, valueFactory: static (_, _, _) => (Half)0.25f);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_FP16_NCHW(
            "input",
            "output",
            new Size(6, 6),
            padValue: 0.5f,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 1, 6, 6], output.GetTensorTypeAndShape().Shape);
        ReadOnlySpan<Half> data = MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>());
        Assert.Equal((Half)0.5f, data[0]);
        Assert.Equal((Half)0.25f, data[1 * 6 + 0]);
        Assert.Equal((Half)0.25f, data[3 * 6 + 5]);
        Assert.Equal((Half)0.5f, data[5 * 6 + 5]);
        await SavePadResizeFP16VisualArtifacts(executionBackend);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task Rotate90FP32NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP32NchwTensor(channels: 2, height: 2, width: 3, valueFactory: static (channel, y, x) => channel * 1000 + y * 10 + x);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_FP32_NCHW(
            "input",
            "output",
            Rotate90Mode.Clockwise90,
            outputTransformKey: "rotateToSource",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal([1, 2, 3, 2], output.GetTensorTypeAndShape().Shape);
        Assert.Equal(
            [10, 0, 11, 1, 12, 2, 1010, 1000, 1011, 1001, 1012, 1002],
            output.GetTensorDataAsSpan<float>().ToArray());
        Assert.True(context.TryGet("rotateToSource", out ICoordinateBackTransform _));
        await SaveRotate90FP32VisualArtifacts(executionBackend);
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task Rotate90FP16NCHW_ReturnsExpectedTensor(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using OrtValue input = CreateFP16NchwTensor(channels: 2, height: 2, width: 3, valueFactory: static (channel, y, x) => (Half)(channel * 1000 + y * 10 + x));
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_FP16_NCHW(
            "input",
            "output",
            Rotate90Mode.Clockwise90,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        Assert.Equal([1, 2, 3, 2], output.GetTensorTypeAndShape().Shape);
        Assert.Equal(
            [(Half)10, (Half)0, (Half)11, (Half)1, (Half)12, (Half)2, (Half)1010, (Half)1000, (Half)1011, (Half)1001, (Half)1012, (Half)1002],
            MemoryMarshal.Cast<Float16, Half>(output.GetTensorDataAsSpan<Float16>()).ToArray());
        await SaveRotate90FP16VisualArtifacts(executionBackend);
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

    static Task SaveResizeFP32VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP32VisualCase(
            executionBackend,
            "Op_Onnx_Resize_FP32_NCHW",
            "visual_gradient_800x600_to_1000x750",
            width: 800,
            height: 600,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize_FP32_NCHW(
                inputKey,
                outputKey,
                new Size(1000, 750),
                isFinal: true,
                executionBackend: backend));

    static Task SaveResizeFP16VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP16VisualCase(
            executionBackend,
            "Op_Onnx_Resize_FP16_NCHW",
            "visual_gradient_800x600_to_1000x750",
            width: 800,
            height: 600,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize_FP16_NCHW(
                inputKey,
                outputKey,
                new Size(1000, 750),
                isFinal: true,
                executionBackend: backend));

    static Task SavePadResizeFP32VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP32VisualCase(
            executionBackend,
            "Op_Onnx_PadResize_FP32_NCHW",
            "visual_gradient_1024x512_to_1024x768",
            width: 1024,
            height: 512,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_FP32_NCHW(
                inputKey,
                outputKey,
                new Size(1024, 768),
                isFinal: true,
                executionBackend: backend));

    static Task SavePadResizeFP16VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP16VisualCase(
            executionBackend,
            "Op_Onnx_PadResize_FP16_NCHW",
            "visual_gradient_1024x512_to_1024x768",
            width: 1024,
            height: 512,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_FP16_NCHW(
                inputKey,
                outputKey,
                new Size(1024, 768),
                isFinal: true,
                executionBackend: backend));

    static Task SaveRotate90FP32VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP32VisualCase(
            executionBackend,
            "Op_Onnx_Rotate90_FP32_NCHW",
            "visual_gradient_800x600_clockwise",
            width: 800,
            height: 600,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_FP32_NCHW(
                inputKey,
                outputKey,
                Rotate90Mode.Clockwise90,
                isFinal: true,
                executionBackend: backend));

    static Task SaveRotate90FP16VisualArtifacts(InferenceBackend executionBackend) =>
        ExecuteFP16VisualCase(
            executionBackend,
            "Op_Onnx_Rotate90_FP16_NCHW",
            "visual_gradient_800x600_clockwise",
            width: 800,
            height: 600,
            static (inputKey, outputKey, backend) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_FP16_NCHW(
                inputKey,
                outputKey,
                Rotate90Mode.Clockwise90,
                isFinal: true,
                executionBackend: backend));

    static async Task ExecuteFP32VisualCase(
        InferenceBackend executionBackend,
        string operationName,
        string caseName,
        int width,
        int height,
        Func<string, string, InferenceBackend, global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase> instructionFactory)
    {
        using OrtValue input = CreateFP32NchwTensor(3, height, width, CreateVisualGradientValue);
        await using var context = VmRunContextFactory.Create();
        using global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase instruction = instructionFactory("visual.input", "visual.output", executionBackend);

        context.Set("visual.input", input);
        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        SaveFP32Artifacts(input, context.Get<OrtValue>("visual.output"), executionBackend, operationName, caseName);
    }

    static async Task ExecuteFP16VisualCase(
        InferenceBackend executionBackend,
        string operationName,
        string caseName,
        int width,
        int height,
        Func<string, string, InferenceBackend, global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase> instructionFactory)
    {
        using OrtValue input = CreateFP16NchwTensor(3, height, width, static (channel, y, x) => (Half)CreateVisualGradientValue(channel, y, x));
        await using var context = VmRunContextFactory.Create();
        using global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase instruction = instructionFactory("visual.input", "visual.output", executionBackend);

        context.Set("visual.input", input);
        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        SaveFP16Artifacts(input, context.Get<OrtValue>("visual.output"), executionBackend, operationName, caseName);
    }

    static float CreateVisualGradientValue(int channel, int y, int x) =>
        channel switch
        {
            0 => (x % 256) / 255f,
            1 => (y % 256) / 255f,
            2 => ((x + y) % 256) / 255f,
            _ => 0f
        };

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

    static void AssertChannelConstant(ReadOnlySpan<float> data, int channels, int height, int width, ReadOnlySpan<float> expectedValues)
    {
        for(int channel = 0; channel < channels; channel++)
        {
            ReadOnlySpan<float> channelData = data.Slice(channel * height * width, height * width);
            foreach(float value in channelData)
                Assert.Equal(expectedValues[channel], value, precision: 5);
        }
    }

    static void AssertChannelConstant(ReadOnlySpan<Half> data, int channels, int height, int width, ReadOnlySpan<Half> expectedValues)
    {
        for(int channel = 0; channel < channels; channel++)
        {
            ReadOnlySpan<Half> channelData = data.Slice(channel * height * width, height * width);
            foreach(Half value in channelData)
                Assert.Equal(expectedValues[channel], value);
        }
    }

    static void SaveFP32Artifacts(
        OrtValue input,
        OrtValue output,
        InferenceBackend executionBackend,
        string operationName,
        string caseName)
    {
        using Mat source = CreatePreviewFromFP32Nchw(input);
        using Mat actual = CreatePreviewFromFP32Nchw(output);
        SaveArtifacts(source, actual, executionBackend, operationName, caseName);
    }

    static void SaveFP16Artifacts(
        OrtValue input,
        OrtValue output,
        InferenceBackend executionBackend,
        string operationName,
        string caseName)
    {
        using Mat source = CreatePreviewFromFP16Nchw(input);
        using Mat actual = CreatePreviewFromFP16Nchw(output);
        SaveArtifacts(source, actual, executionBackend, operationName, caseName);
    }

    static void SaveArtifacts(
        Mat source,
        Mat actual,
        InferenceBackend executionBackend,
        string operationName,
        string caseName)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} source: {caseName}",
            source,
            environment.CreateOperationArtifactPath(
                "Onnx",
                operationName,
                executionBackend.ToString(),
                caseName,
                "source"));

        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} actual: {caseName}",
            actual,
            environment.CreateOperationArtifactPath(
                "Onnx",
                operationName,
                executionBackend.ToString(),
                caseName,
                "actual"));
    }

    static Mat CreatePreviewFromFP32Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(
            tensor.GetTensorDataAsSpan<float>(),
            checked((int)shape[1]),
            checked((int)shape[2]),
            checked((int)shape[3]),
            static value => NormalizePreviewByte(value));
    }

    static Mat CreatePreviewFromFP16Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(
            MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorDataAsSpan<Float16>()),
            checked((int)shape[1]),
            checked((int)shape[2]),
            checked((int)shape[3]),
            static value => NormalizePreviewByte((float)value));
    }

    static Mat CreatePreviewFromNchw<T>(
        ReadOnlySpan<T> data,
        int channels,
        int height,
        int width,
        Func<T, byte> convert)
    {
        var image = new Mat(height, width, MatType.CV_8UC3);

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                byte blue = ReadChannel(data, channels, height, width, channel: 0, y, x, convert);
                byte green = ReadChannel(data, channels, height, width, channel: 1, y, x, convert);
                byte red = ReadChannel(data, channels, height, width, channel: 2, y, x, convert);
                image.Set(y, x, new Vec3b(blue, green, red));
            }
        }

        return image;
    }

    static byte ReadChannel<T>(
        ReadOnlySpan<T> data,
        int channels,
        int height,
        int width,
        int channel,
        int y,
        int x,
        Func<T, byte> convert)
    {
        int sourceChannel = Math.Min(channel, channels - 1);
        int index = sourceChannel * height * width + y * width + x;
        return convert(data[index]);
    }

    static byte NormalizePreviewByte(float value)
    {
        float scaled = value is >= 0f and <= 1f
            ? value * 255f
            : value % 256f;

        if(scaled < 0)
            scaled += 256f;

        return (byte)Math.Clamp((int)MathF.Round(scaled), 0, 255);
    }
}
