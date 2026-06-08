using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Owns ONNX contexts and VM resources used by the pipeline Avalonia lab.
/// </summary>
/// <remarks>
/// The old lab kept runners directly in the frame loop. The VM lab keeps them as named resources so instructions can
/// call model services without owning model lifetime or batching policy.
/// </remarks>
internal sealed class PipelineModelResources : IDisposable
{
    readonly RealTimeAvaloniaSettings settings;
    readonly AvaloniaJsonConfig jsonConfig;
    readonly IOnnxAssetResolver assetResolver;
    bool disposed;

    PipelineModelResources(
        RealTimeAvaloniaSettings settings,
        AvaloniaJsonConfig jsonConfig,
        IOnnxAssetResolver assetResolver)
    {
        this.settings = settings;
        this.jsonConfig = jsonConfig;
        this.assetResolver = assetResolver;
    }

    public OnnxRuntimeContext ModelBox { get; private set; } = null!;
    public OnnxRuntimeContext ModelObb { get; private set; } = null!;
    public OnnxRuntimeContext ModelSeg { get; private set; } = null!;
    public OnnxRuntimeContext ModelCls { get; private set; } = null!;
    public OnnxRuntimeContext ModelPose { get; private set; } = null!;
    public OnnxRuntimeContext ModelDet { get; private set; } = null!;
    public OnnxRuntimeContext ModelRec { get; private set; } = null!;

    public OnnxRunnerResource<Mat, IDetectionResult<YoloBox>> BoxResource { get; private set; } = null!;
    public YoloObbBatchedResource ObbResource { get; private set; } = null!;
    public OnnxRunnerResource<Mat, IBatchedResult> SegResource { get; private set; } = null!;
    public OnnxRunnerResource<Mat, IBatchedResult> ClsResource { get; private set; } = null!;
    public OnnxRunnerResource<Mat, IDetectionResult<YoloPose>> PoseResource { get; private set; } = null!;
    public OnnxRunnerResource<Mat, Mat> DetResource { get; private set; } = null!;
    public ReloadableOnnxRunnerResource<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>> RecResource { get; private set; } = null!;

    public static async Task<PipelineModelResources> CreateAsync(
        RealTimeAvaloniaSettings settings,
        RecognitionOptions recognitionOptions,
        AvaloniaJsonConfig jsonConfig,
        IOnnxAssetResolver assetResolver)
    {
        var resources = new PipelineModelResources(settings, jsonConfig, assetResolver);
        await resources.InitializeAsync(recognitionOptions).ConfigureAwait(false);
        return resources;
    }

    public void EnsureDetFrameShape(Mat letterboxedFrame)
    {
        ArgumentNullException.ThrowIfNull(letterboxedFrame);

        if(ModelDet.IsInputPersistentValueInitialized(ModelDet.Model.PrimaryInputName))
            return;

        // PaddleOCR Det is dynamic by spatial dimensions. Initialize persistent buffers from the actual tensor shape
        // that the VM produces, not from a duplicated config value.
        ModelDet.InitInputPersistentValue(ModelDet.Model.PrimaryInputName, [1, 3, letterboxedFrame.Width, letterboxedFrame.Height]);
        ModelDet.InitOutputPersistentValue(ModelDet.Model.PrimaryOutputName, [1, 1, letterboxedFrame.Width, letterboxedFrame.Height]);
    }

    public async ValueTask EnsureRecognitionBatchAsync(RecognitionOptions recognitionOptions, CancellationToken cancellationToken)
    {
        ModelRec.InitInputPersistentValue(
            ModelRec.Model.PrimaryInputName,
            [recognitionOptions.BatchSize, 3, recognitionOptions.RecognitionInputHeight, recognitionOptions.RecognitionInputWidth]);

        ModelRec.InitOutputPersistentValue(
            ModelRec.Model.PrimaryOutputName,
            [recognitionOptions.BatchSize, recognitionOptions.RecognitionOutputItemCount, ResolveRecognitionOutputAttributes()]);

        var runner = new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(ModelRec);
        await RecResource.ReplaceRunnerAsync(runner, cancellationToken).ConfigureAwait(false);
    }

    async Task InitializeAsync(RecognitionOptions recognitionOptions)
    {
        OnnxRuntimeModelSource modelBoxSource = await ResolveDemoModelAsync(ModelNaming.GetFileName(settings.BoxModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource modelObbSource = await ResolveDemoModelAsync(ModelNaming.GetFileName(settings.ObbModelName, settings.InputSize, settings.ObbBatchSize, settings.ObbBatchPrecision, isByteBgr: settings.ObbBatchUseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource modelSegSource = await ResolveDemoModelAsync(ModelNaming.GetFileName(settings.SegModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource modelClsSource = await ResolveDemoModelAsync(ModelNaming.GetFileName(settings.ClsModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource modelPoseSource = await ResolveDemoModelAsync(ModelNaming.GetFileName(settings.PoseModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource paddleDetModelSource = await ResolveDemoModelAsync(GetPaddleModelPath("/paddleocr/detection/v5/det.onnx", settings.PaddleDetModelPrecision, settings.PaddleDetUseByteBgr)).ConfigureAwait(false);
        OnnxRuntimeModelSource recognitionModelSource = await ResolvePaddleRecModelSourceAsync().ConfigureAwait(false);
        if(string.IsNullOrWhiteSpace(settings.PaddleRecModelPath))
            await ResolveDemoModelAsync("/paddleocr/languages/english/dict.txt").ConfigureAwait(false);

        ModelBox = new OnnxRuntimeContext(modelBoxSource, settings.InferenceBackend);
        ModelObb = new OnnxRuntimeContext(modelObbSource, settings.InferenceBackend);
        ModelSeg = new OnnxRuntimeContext(modelSegSource, settings.InferenceBackend);
        ModelCls = new OnnxRuntimeContext(modelClsSource, settings.InferenceBackend);
        ModelPose = new OnnxRuntimeContext(modelPoseSource, settings.InferenceBackend);
        ModelDet = new OnnxRuntimeContext(paddleDetModelSource, settings.PaddleDetInferenceBackend);
        ModelRec = new OnnxRuntimeContext(recognitionModelSource, settings.PaddleRecInferenceBackend);

        IRunner<Mat, IDetectionResult<YoloBox>> boxRunner = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(ModelBox);
        IRunner<List<Mat>, IDetectionResult<YoloObb>> obbRunner = CreateObbBatchRunner(ModelObb);
        obbRunner.OutAs<IExtractorThreshold>()!.Threshold = jsonConfig.Inference.ObbThreshold;

        BoxResource = new OnnxRunnerResource<Mat, IDetectionResult<YoloBox>>("yolo.box", boxRunner);
        ObbResource = new YoloObbBatchedResource(
            "yolo.obb",
            obbRunner,
            new OnnxBatchedResourceOptions(
                MaxBatchSize: settings.ObbBatchSize,
                MaxWaitTime: TimeSpan.FromMilliseconds(settings.ObbBatchWaitMilliseconds),
                MaxPendingRequests: settings.ObbBatchSize * 8));
        SegResource = new OnnxRunnerResource<Mat, IBatchedResult>("yolo.seg", YoloSegFactory.CreateRunner(ModelSeg));
        ClsResource = new OnnxRunnerResource<Mat, IBatchedResult>("yolo.cls", YoloClsFactory.CreateRunner(ModelCls));
        PoseResource = new OnnxRunnerResource<Mat, IDetectionResult<YoloPose>>("yolo.pose", YoloPoseFactory.CreateRunner(ModelPose));
        DetResource = new OnnxRunnerResource<Mat, Mat>("paddle.det", PaddleOCRDetFactory.CreateRunner<Mat, Mat>(ModelDet, MatType.CV_32FC1));

        ModelRec.InitInputPersistentValue(
            ModelRec.Model.PrimaryInputName,
            [recognitionOptions.BatchSize, 3, recognitionOptions.RecognitionInputHeight, recognitionOptions.RecognitionInputWidth]);
        ModelRec.InitOutputPersistentValue(
            ModelRec.Model.PrimaryOutputName,
            [recognitionOptions.BatchSize, recognitionOptions.RecognitionOutputItemCount, ResolveRecognitionOutputAttributes()]);
        var recRunner = new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(ModelRec);
        RecResource = new ReloadableOnnxRunnerResource<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>>(
            "paddle.rec",
            recRunner,
            disposeReplacedRunner: false);
    }

    static IRunner<List<Mat>, IDetectionResult<YoloObb>> CreateObbBatchRunner(OnnxRuntimeContext model)
    {
        var inputMeta = model.Model.Session.InputMetadata.Values.First();
        return inputMeta.ElementDataType switch
        {
            Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float => YoloObbFactory.List_PosCvdnn_FP32(model),
            _ => throw new NotSupportedException($"Batch OBB runner does not support input type {inputMeta.ElementDataType}. Use ObbBatchPrecision=fp32 and ObbBatchUseByteBgr=false.")
        };
    }

    int ResolveRecognitionOutputAttributes()
    {
        long[] outputShape = ModelRec.Model.ModelOutputShapes[ModelRec.Model.PrimaryOutputName];
        if(outputShape.Length < 3)
            throw new InvalidOperationException($"PaddleOCR Rec output must be 3D, actual shape: {string.Join(",", outputShape)}");

        long attributes = outputShape[2];
        if(attributes <= 0)
            throw new InvalidOperationException($"PaddleOCR Rec output attributes dimension is dynamic or invalid: {string.Join(",", outputShape)}");

        return checked((int)attributes);
    }

    static string GetPaddleModelPath(string modelPath, string precision, bool isByteBgr)
    {
        if(string.Equals(precision, "fp32", StringComparison.OrdinalIgnoreCase) && !isByteBgr)
            return modelPath;

        string modelDirectory = Path.GetDirectoryName(modelPath)?.Replace('\\', '/') ?? string.Empty;
        string modelFileName = Path.GetFileNameWithoutExtension(modelPath);
        string modelExtension = Path.GetExtension(modelPath);
        string precisionSuffix = string.Equals(precision, "fp32", StringComparison.OrdinalIgnoreCase) ? string.Empty : $"_{precision}";
        string byteBgrSuffix = isByteBgr ? "_bytebgr" : string.Empty;
        string configuredModelFileName = $"{modelFileName}{precisionSuffix}{byteBgrSuffix}{modelExtension}";

        return string.IsNullOrEmpty(modelDirectory)
            ? configuredModelFileName
            : $"{modelDirectory}/{configuredModelFileName}";
    }

    ValueTask<OnnxRuntimeModelSource> ResolveDemoModelAsync(string modelId) =>
        assetResolver.ResolveModelSourceAsync(modelId);

    async Task<OnnxRuntimeModelSource> ResolvePaddleRecModelSourceAsync()
    {
        if(string.IsNullOrWhiteSpace(settings.PaddleRecModelPath))
            return await ResolveDemoModelAsync(GetPaddleModelPath("/paddleocr/languages/english/rec.onnx", settings.PaddleRecModelPrecision, settings.PaddleRecUseByteBgr)).ConfigureAwait(false);

        if(!Path.IsPathRooted(settings.PaddleRecModelPath))
            throw new InvalidOperationException($"PaddleRecModelPath must be a full file path: {settings.PaddleRecModelPath}");

        string modelPath = Path.GetFullPath(settings.PaddleRecModelPath);
        if(!File.Exists(modelPath))
            throw new FileNotFoundException("Configured PaddleOCR Rec model was not found.", modelPath);

        return OnnxRuntimeModelSource.FromFile(modelPath);
    }

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;
        BoxResource?.Dispose();
        ObbResource?.Dispose();
        SegResource?.Dispose();
        ClsResource?.Dispose();
        PoseResource?.Dispose();
        DetResource?.Dispose();
        RecResource?.Dispose();
    }
}
