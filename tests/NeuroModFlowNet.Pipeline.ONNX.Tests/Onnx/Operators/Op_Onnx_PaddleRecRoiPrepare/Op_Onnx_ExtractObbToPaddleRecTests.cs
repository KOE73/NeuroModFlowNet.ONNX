using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_PaddleRecRoiPrepare;

public sealed class Op_Onnx_ExtractObbToPaddleRecTests
{
    public static IEnumerable<object[]> SupportedBackends =>
        OnnxExecutionBackendMatrix.EnabledBackends
            .Where(static backend => backend is InferenceBackend.Cpu or InferenceBackend.Cuda)
            .Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(SupportedBackends))]
    public async Task ExtractObbToPaddleRecFP32NCHW_ProducesPaddleNormalizedBatch(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        const int sourceWidth = 16;
        const int sourceHeight = 8;
        const int channels = 3;
        YoloObb[] boxes =
        [
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight)
        ];

        (OrtValue output, int? actualCount, VmRunContext context) =
            await ExecuteExtractAsync(executionBackend, sourceWidth, sourceHeight, boxes);
        await using(context)
        {
            var outputInfo = output.GetTensorTypeAndShape();
            Assert.Equal(TensorElementType.Float, outputInfo.ElementDataType);
            Assert.Equal([boxes.Length, channels, sourceHeight, sourceWidth], outputInfo.Shape);
            Assert.Null(actualCount);

            ReadOnlySpan<float> outputData = output.GetTensorDataAsSpan<float>();
            for(int index = 0; index < outputData.Length; index++)
                Assert.InRange(outputData[index], -0.00001f, 0.00001f);
        }
    }

    [Theory]
    [MemberData(nameof(SupportedBackends))]
    public async Task ExtractObbToPaddleRecFP32NCHW_FixedMaxRoiCountWritesCapacityBatchAndActualCount(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        const int sourceWidth = 16;
        const int sourceHeight = 8;
        const int channels = 3;
        const int maxRoiCount = 6;
        YoloObb[] boxes =
        [
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight)
        ];

        (OrtValue output, int? actualCount, VmRunContext context) =
            await ExecuteExtractAsync(executionBackend, sourceWidth, sourceHeight, boxes, maxRoiCount: maxRoiCount, actualCountOutputKey: "paddle.rec.count");
        await using(context)
        {
            Assert.Equal(boxes.Length, actualCount);
            Assert.Equal([maxRoiCount, channels, sourceHeight, sourceWidth], output.GetTensorTypeAndShape().Shape);
        }
    }

    [Fact]
    public async Task ExtractObbToPaddleRecFP32NCHW_FixedMaxRoiCountAllowsZeroActualRoi()
    {
        const int sourceWidth = 16;
        const int sourceHeight = 8;
        const int channels = 3;
        const int maxRoiCount = 6;

        (OrtValue output, int? actualCount, VmRunContext context) =
            await ExecuteExtractAsync(InferenceBackend.Cpu, sourceWidth, sourceHeight, [], maxRoiCount: maxRoiCount, actualCountOutputKey: "paddle.rec.count");
        await using(context)
        {
            Assert.Equal(0, actualCount);
            Assert.Equal([maxRoiCount, channels, sourceHeight, sourceWidth], output.GetTensorTypeAndShape().Shape);
        }
    }

    [Fact]
    public async Task ExtractObbToPaddleRecFP32NCHW_FixedMaxRoiCountTruncatesOnlyWhenConfigured()
    {
        const int sourceWidth = 16;
        const int sourceHeight = 8;
        const int maxRoiCount = 2;
        YoloObb[] boxes =
        [
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight)
        ];

        (OrtValue output, int? actualCount, VmRunContext context) =
            await ExecuteExtractAsync(
                InferenceBackend.Cpu,
                sourceWidth,
                sourceHeight,
                boxes,
                maxRoiCount: maxRoiCount,
                overflowPolicy: PaddleRecRoiOverflowPolicy.Truncate,
                actualCountOutputKey: "paddle.rec.count");
        await using(context)
        {
            Assert.Equal(maxRoiCount, actualCount);
            Assert.Equal(maxRoiCount, output.GetTensorTypeAndShape().Shape[0]);
        }
    }

    [Fact]
    public async Task ExtractObbToPaddleRecFP32NCHW_FixedMaxRoiCountFailsOverflowByDefault()
    {
        const int sourceWidth = 16;
        const int sourceHeight = 8;
        YoloObb[] boxes =
        [
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight),
            CreateWholeImageObb(sourceWidth, sourceHeight)
        ];

        using OrtValue source = CreateFP32NchwTensor(channels: 3, sourceHeight, sourceWidth, 0.5f);
        await using var context = VmRunContextFactory.Create();
        context.Set("image.gpu", source);
        context.Set("text.obb", boxes);

        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW(
            "image.gpu",
            "text.obb",
            "paddle.rec.input",
            new Size(sourceWidth, sourceHeight),
            maxRoiCount: 2,
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Fail, result.Kind);
    }

    [Fact]
    public void ExtractObbToPaddleRecFP32NCHW_RejectsInitializerAlgorithmsForDynamicObb()
    {
        Assert.Throws<NotSupportedException>(() => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW(
            "image",
            "obb",
            "output",
            new Size(320, 48),
            algorithm: PaddleRecRoiPrepareAlgorithm.PerRegionGridSampleConcat));
    }

    static async Task<(OrtValue Output, int? ActualCount, VmRunContext Context)> ExecuteExtractAsync(
        InferenceBackend executionBackend,
        int sourceWidth,
        int sourceHeight,
        YoloObb[] boxes,
        int? maxRoiCount = null,
        PaddleRecRoiOverflowPolicy overflowPolicy = PaddleRecRoiOverflowPolicy.Fail,
        string? actualCountOutputKey = null)
    {
        const int channels = 3;
        OrtValue source = CreateFP32NchwTensor(channels, sourceHeight, sourceWidth, 0.5f);
        var context = VmRunContextFactory.Create();
        context.Set("image.host", source, disposeWithContext: true);

        if(executionBackend == InferenceBackend.Cuda)
        {
            var upload = new Copy_OrtTensor_To_ModelDevice("image.host", "image.gpu", executionBackend);
            context.AddOwnedResource(upload);
            OpResult uploadResult = await upload.ExecuteAsync(context, CancellationToken.None);
            Assert.Equal(OpResultKind.Continue, uploadResult.Kind);
        }
        else
        {
            context.Set("image.gpu", source);
        }

        context.Set("text.obb", boxes);

        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW(
            "image.gpu",
            "text.obb",
            "paddle.rec.input",
            new Size(sourceWidth, sourceHeight),
            paddingPixels: 0,
            paddingScale: 0,
            maxRoiCount: maxRoiCount,
            overflowPolicy: overflowPolicy,
            actualCountOutputKey: actualCountOutputKey,
            isFinal: true,
            executionBackend: executionBackend);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        return (
            context.Get<OrtValue>("paddle.rec.input"),
            actualCountOutputKey is null ? null : context.Get<int>(actualCountOutputKey),
            context);
    }

    static OrtValue CreateFP32NchwTensor(int channels, int height, int width, float value)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [1, channels, height, width]);

        tensor.GetTensorMutableDataAsSpan<float>().Fill(value);
        return tensor;
    }

    static YoloObb CreateWholeImageObb(int width, int height) =>
        new()
        {
            X = (width - 1) * 0.5f,
            Y = (height - 1) * 0.5f,
            W = width - 1,
            H = height - 1,
            Angle = 0,
            Score = 1,
            Class = 0
        };
}
