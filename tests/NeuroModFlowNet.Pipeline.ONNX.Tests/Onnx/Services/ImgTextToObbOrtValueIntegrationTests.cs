using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Services;

public sealed class ImgTextToObbOrtValueIntegrationTests
{
    const string ModelPath = "models/img-text-to-obb/img-text-to-obb__640_b4_fp32.onnx";

    [Fact]
    public async Task TwoVmRuns_PrepareImagesBatch4ModelAndReceiveOwnObbResults()
    {
        if(!File.Exists(ModelPath))
            return;

        string modelPath = Path.GetFullPath(ModelPath);

        await using var endpoint = new OrtValueBatchedInferenceEndpoint<YoloObb>(
            name: "img-text-to-obb.fp32.b4",
            modelPath: modelPath,
            executionBackend: InferenceBackend.Cpu,
            options: new OnnxBatchedResourceOptions(MaxBatchSize: 2, MaxWaitTime: TimeSpan.FromMilliseconds(500), MaxPendingRequests: 4),
            batchInputAssembler: new ConcatOrtValueBatchInputAssembler(fixedBatchSize: 4),
            outputShapeResolver: new YoloNmsOutputShapeResolver(defaultItemCount: 300, fieldCount: 7),
            outputDecoder: new YoloObbOrtValueOutputDecoder(scoreThreshold: 0.25f));

        var inference = new Model_OrtValueInference<YoloObb>(
            name: "Model_ImgTextToObb_OrtValue_FP32_B4",
            inputKey: "image.modelInput",
            outputKey: "obb.results",
            endpoint: endpoint);

        await using VmRunContext firstContext = await PrepareContext("for_tests.png");
        await using VmRunContext secondContext = await PrepareContext("for_tests_perspective_real_0.png");

        ValueTask<OpResult> firstTask = inference.ExecuteAsync(firstContext, CancellationToken.None);
        ValueTask<OpResult> secondTask = inference.ExecuteAsync(secondContext, CancellationToken.None);

        Assert.Equal(OpResult.Continue, await firstTask);
        Assert.Equal(OpResult.Continue, await secondTask);

        YoloObb[] firstResult = firstContext.Get<YoloObb[]>("obb.results");
        YoloObb[] secondResult = secondContext.Get<YoloObb[]>("obb.results");

        Assert.NotNull(firstResult);
        Assert.NotNull(secondResult);
        Assert.All(firstResult, static box => Assert.InRange(box.Score, 0.25f, 1f));
        Assert.All(secondResult, static box => Assert.InRange(box.Score, 0.25f, 1f));
    }

    static async Task<VmRunContext> PrepareContext(string imageFileName)
    {
        using Mat source = RealImageTestSource.LoadBgr(imageFileName);
        using Mat sourceClone = source.Clone();
        OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(sourceClone);

        var context = VmRunContextFactory.Create();
        context.Set("image.input", input, disposeWithContext: true);

        using var padResize = new Op_Onnx_PadResize_U8_NHWC(
            inputKey: "image.input",
            outputKey: "image.resized",
            targetSize: new CvSize(640, 640),
            outputTransformKey: "image.resized.toSource",
            stride: 32,
            mode: PadResizeMode.FixedCanvas,
            isFinal: false,
            executionBackend: InferenceBackend.Cpu);

        using var prepare = new Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
            inputKey: "image.resized",
            outputKey: "image.modelInput",
            isFinal: false,
            executionBackend: InferenceBackend.Cpu);

        Assert.Equal(OpResult.Continue, await padResize.ExecuteAsync(context, CancellationToken.None));
        Assert.Equal(OpResult.Continue, await prepare.ExecuteAsync(context, CancellationToken.None));

        return context;
    }
}
