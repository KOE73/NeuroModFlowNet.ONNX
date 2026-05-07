using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Owns ONNX contexts and runners used by the realtime Avalonia inference loop.
/// RU: Владеет ONNX contexts и runners, которые используются realtime Avalonia inference loop.
/// </summary>
/// <remarks>
/// EN: This class loads model assets, creates box, OBB, and PaddleOCR Rec runners, and rebuilds PaddleOCR Rec persistent
/// input/output buffers when batch or recognition tensor size changes.
/// RU: Класс загружает model assets, создает box, OBB и PaddleOCR Rec runners, а также пересоздает PaddleOCR Rec
/// persistent input/output buffers при изменении batch или размера recognition tensor.
/// </remarks>
internal sealed class InferenceResources : IDisposable
{
    readonly RealTimeAvaloniaSettings settings;

    InferenceResources(RealTimeAvaloniaSettings settings)
    {
        this.settings = settings;
    }

    public OnnxRuntimeContext ModelBox { get; private set; } = null!;
    public OnnxRuntimeContext ModelObb { get; private set; } = null!;
    public OnnxRuntimeContext ModelSeg { get; private set; } = null!;
    public OnnxRuntimeContext ModelCls { get; private set; } = null!;
    public OnnxRuntimeContext ModelPose { get; private set; } = null!;
    public OnnxRuntimeContext ModelRec { get; private set; } = null!;
    public IRunner<Mat, IDetectionResult<YoloBox>> RunnerBox { get; private set; } = null!;
    public IRunner<Mat, IDetectionResult<YoloObb>> RunnerObb { get; private set; } = null!;
    public IRunner<Mat, IBatchedResult> RunnerSeg { get; private set; } = null!;
    public IRunner<Mat, IBatchedResult> RunnerCls { get; private set; } = null!;
    public IRunner<Mat, IDetectionResult<YoloPose>> RunnerPose { get; private set; } = null!;
    public ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>? RunnerRec { get; private set; }
    public IReadOnlyList<RuntimeModelInfo> ModelInfos { get; private set; } = [];

    public static async Task<InferenceResources> CreateAsync(
        RealTimeAvaloniaSettings settings,
        RecognitionOptions recognitionOptions)
    {
        var resources = new InferenceResources(settings);
        await resources.InitializeAsync(recognitionOptions);
        return resources;
    }

    public void EnsureRecognitionBatch(RecognitionOptions recognitionOptions)
    {
        ModelRec.InitInputPersistentValue(
            ModelRec.PrimaryInputName,
            [recognitionOptions.BatchSize, 3, recognitionOptions.RecognitionInputHeight, recognitionOptions.RecognitionInputWidth]);

        ModelRec.InitOutputPersistentValue(
            ModelRec.PrimaryOutputName,
            [recognitionOptions.BatchSize, recognitionOptions.RecognitionOutputItemCount, ResolveRecognitionOutputAttributes()]);

        RunnerRec?.Dispose();
        RunnerRec = new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(ModelRec);
        RefreshModelInfos();
    }

    async Task InitializeAsync(RecognitionOptions recognitionOptions)
    {
        string modelBoxPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(settings.BoxModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr));
        string modelObbPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(settings.ObbModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr));
        string modelSegPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(settings.SegModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr));
        string modelClsPath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(settings.ClsModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr));
        string modelPosePath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName(settings.PoseModelName, settings.InputSize, 1, settings.ModelPrecision, isByteBgr: settings.UseByteBgr));
        string recognitionModelPath = await AssetsManager.GetAssetPathAsync(GetPaddleModelPath("/paddleocr/languages/english/rec.onnx", settings.PaddleRecModelPrecision, settings.PaddleRecUseByteBgr));
        await AssetsManager.GetAssetPathAsync("/paddleocr/languages/english/dict.txt");

        ModelBox = new OnnxRuntimeContext(modelBoxPath, settings.InferenceBackend);
        ModelObb = new OnnxRuntimeContext(modelObbPath, settings.InferenceBackend);
        ModelSeg = new OnnxRuntimeContext(modelSegPath, settings.InferenceBackend);
        ModelCls = new OnnxRuntimeContext(modelClsPath, settings.InferenceBackend);
        ModelPose = new OnnxRuntimeContext(modelPosePath, settings.InferenceBackend);
        ModelRec = new OnnxRuntimeContext(recognitionModelPath, settings.PaddleRecInferenceBackend);

        RunnerBox = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(ModelBox);
        RunnerObb = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(ModelObb);
        RunnerSeg = YoloSegFactory.CreateRunner(ModelSeg);
        RunnerCls = YoloClsFactory.CreateRunner(ModelCls);
        RunnerPose = YoloPoseFactory.CreateRunner(ModelPose);
        RunnerObb.OutAs<IExtractorThreshold>()!.Threshold = 0.3f;
        EnsureRecognitionBatch(recognitionOptions);
    }

    void RefreshModelInfos()
    {
        ModelInfos =
        [
            CreateModelInfo("ocr", "OCR", ModelObb, ModelRec),
            CreateModelInfo("box", "Detection Box", ModelBox),
            CreateModelInfo("obb", "OBB Detection", ModelObb),
            CreateModelInfo("seg", "Segmentation", ModelSeg),
            CreateModelInfo("cls", "Classification", ModelCls),
            CreateModelInfo("pose", "Pose", ModelPose),
        ];
    }

    static RuntimeModelInfo CreateModelInfo(string key, string title, OnnxRuntimeContext model)
    {
        string details = string.Join(
            Environment.NewLine,
            model.ModelInputShapes.Select(item => FormatIoLine("IN ", item.Key, ResolveInputType(model, item.Key), ResolveInputShape(model, item.Key)))
                .Concat(model.ModelOutputShapes.Select(item => FormatIoLine("OUT", item.Key, ResolveOutputType(model, item.Key), ResolveOutputShape(model, item.Key)))));

        return new RuntimeModelInfo(key, title, details);
    }

    static RuntimeModelInfo CreateModelInfo(string key, string title, OnnxRuntimeContext detectorModel, OnnxRuntimeContext recognitionModel)
    {
        string details = string.Join(
            Environment.NewLine,
            FormatIoLine("DETIN", detectorModel.PrimaryInputName, ResolveInputType(detectorModel, detectorModel.PrimaryInputName), ResolveInputShape(detectorModel, detectorModel.PrimaryInputName)),
            FormatIoLine("D_OUT", detectorModel.PrimaryOutputName, ResolveOutputType(detectorModel, detectorModel.PrimaryOutputName), ResolveOutputShape(detectorModel, detectorModel.PrimaryOutputName)),
            FormatIoLine("RECIN", recognitionModel.PrimaryInputName, ResolveInputType(recognitionModel, recognitionModel.PrimaryInputName), ResolveInputShape(recognitionModel, recognitionModel.PrimaryInputName)),
            FormatIoLine("R_OUT", recognitionModel.PrimaryOutputName, ResolveOutputType(recognitionModel, recognitionModel.PrimaryOutputName), ResolveOutputShape(recognitionModel, recognitionModel.PrimaryOutputName)));

        return new RuntimeModelInfo(key, title, details);
    }

    static long[] ResolveInputShape(OnnxRuntimeContext model, string name) =>
        model.IsInputPersistentValueInitialized(name) ? model.GetRealInputShape(name) : model.ModelInputShapes[name];

    static long[] ResolveOutputShape(OnnxRuntimeContext model, string name) =>
        model.IsOutputPersistentValueInitialized(name) ? model.GetRealOutputShape(name) : model.ModelOutputShapes[name];

    static string ResolveInputType(OnnxRuntimeContext model, string name) =>
        model.Session.InputMetadata[name].ElementDataType.ToString();

    static string ResolveOutputType(OnnxRuntimeContext model, string name) =>
        model.Session.OutputMetadata[name].ElementDataType.ToString();

    static string FormatIoLine(string marker, string name, string elementType, IReadOnlyList<long> shape) =>
        $"{marker} {ShortName(name),-8} {ShortType(elementType),-7} {FormatShape(shape)}";

    static string ShortName(string name) =>
        name.Length <= 8 ? name : name[..8];

    static string ShortType(string elementType) =>
        elementType.Length <= 7 ? elementType : elementType[..7];

    static string FormatShape(IReadOnlyList<long> shape) =>
        string.Join("x", shape.Select(value => value <= 0 ? "?" : value.ToString()));

    int ResolveRecognitionOutputAttributes()
    {
        long[] outputShape = ModelRec.ModelOutputShapes[ModelRec.PrimaryOutputName];
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

    public void Dispose()
    {
        RunnerBox?.Dispose();
        RunnerObb?.Dispose();
        RunnerSeg?.Dispose();
        RunnerCls?.Dispose();
        RunnerPose?.Dispose();
        RunnerRec?.Dispose();
        ModelBox?.Dispose();
        ModelObb?.Dispose();
        ModelSeg?.Dispose();
        ModelCls?.Dispose();
        ModelPose?.Dispose();
        ModelRec?.Dispose();
    }
}
