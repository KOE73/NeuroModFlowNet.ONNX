using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Model endpoints shared by camera controllers. The detector has one endpoint per model file: each camera strip has
/// its own static model input, cameras with the same model share its endpoint. Text detection and recognition are
/// shared by all cameras. Programs and geometry ops are per controller (ADR-001).
///
/// RU: Endpoint-ы моделей, общие для контроллеров камер. У детектора endpoint на каждый файл модели: у полосы каждой
/// камеры свой статический вход, камеры с одинаковой моделью делят endpoint. Детекция и распознавание текста общие
/// для всех камер. Программы и geometry-ops у каждого контроллера свои (ADR-001).
/// </summary>
internal sealed class SharedInferenceEndpoints : IAsyncDisposable
{
    readonly Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloBox>> boxDetectors;
    readonly Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloObb>> obbDetectors;

    SharedInferenceEndpoints(
        Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloBox>> boxDetectors,
        Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloObb>> obbDetectors,
        OrtValueBatchedInferenceEndpoint<CropTextRegion>? textDetector,
        OrtValueBatchedInferenceEndpoint<PaddleOCRRecExtractor.OcrResult>? recognition)
    {
        this.boxDetectors = boxDetectors;
        this.obbDetectors = obbDetectors;
        TextDetector = textDetector;
        Recognition = recognition;
    }

    /// <summary>Text OBB detector fed with whole crop batches [maxCrops, 3, S, S]; results keep the crop index.</summary>
    public OrtValueBatchedInferenceEndpoint<CropTextRegion>? TextDetector { get; }

    public OrtValueBatchedInferenceEndpoint<PaddleOCRRecExtractor.OcrResult>? Recognition { get; }

    public OrtValueBatchedInferenceEndpoint<YoloBox> GetBoxDetector(string modelPath) => boxDetectors[modelPath];

    public OrtValueBatchedInferenceEndpoint<YoloObb> GetObbDetector(string modelPath) => obbDetectors[modelPath];

    public static SharedInferenceEndpoints Create(MultiCameraConfig config, IReadOnlyList<CameraConfig> cameras)
    {
        ArgumentNullException.ThrowIfNull(config);

        DetectorConfig detector = config.Detector;

        // The CGP bag model does not satisfy strict CUDA (ORT assigns some nodes to CPU EP and fallback is disabled),
        // so the detector backend can differ from the geometry backend; TensorRT is its proven GPU path.
        InferenceBackend detectorBackend = detector.Backend ?? config.Backend;
        var boxDetectors = new Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloBox>>(StringComparer.OrdinalIgnoreCase);
        var obbDetectors = new Dictionary<string, OrtValueBatchedInferenceEndpoint<YoloObb>>(StringComparer.OrdinalIgnoreCase);

        foreach(IGrouping<string, CameraConfig> group in cameras.GroupBy(static camera => camera.Detector.ModelPath, StringComparer.OrdinalIgnoreCase))
        {
            // Static batch-1 models: one request is one frame, no cross-camera batching and no wait.
            var options = new OnnxBatchedResourceOptions(MaxBatchSize: 1, MaxWaitTime: TimeSpan.Zero, MaxPendingRequests: Math.Max(8, group.Count() * 8));
            string name = $"multicamera.detector.{Path.GetFileNameWithoutExtension(group.Key)}";

            if(detector.IsBox)
            {
                boxDetectors[group.Key] = new OrtValueBatchedInferenceEndpoint<YoloBox>(
                    name,
                    group.Key,
                    detectorBackend,
                    options,
                    new SingleOrtValueBatchInputAssembler(),
                    new YoloNmsOutputShapeResolver(defaultItemCount: detector.MaxItemCount, fieldCount: 6),
                    new YoloBoxOrtValueOutputDecoder(detector.ScoreThreshold));
            }
            else
            {
                obbDetectors[group.Key] = new OrtValueBatchedInferenceEndpoint<YoloObb>(
                    name,
                    group.Key,
                    detectorBackend,
                    options,
                    new SingleOrtValueBatchInputAssembler(),
                    new YoloNmsOutputShapeResolver(defaultItemCount: detector.MaxItemCount, fieldCount: 7),
                    new YoloObbOrtValueOutputDecoder(detector.ScoreThreshold));
            }
        }

        if(!config.Ocr.Enabled)
            return new SharedInferenceEndpoints(boxDetectors, obbDetectors, null, null);

        OcrConfig ocr = config.Ocr;

        // One request = one crop batch [maxCrops, 3, S, S] that already matches the fixed model batch, so requests are
        // not merged (MaxBatchSize 1); the request assembler forwards the batch and reports its row count to the decoder.
        var textEndpoint = new OrtValueBatchedInferenceEndpoint<CropTextRegion>(
            name: "multicamera.text.obb.crops",
            modelPath: ocr.TextObbModelPath!,
            executionBackend: config.Backend,
            options: new OnnxBatchedResourceOptions(
                MaxBatchSize: 1,
                MaxWaitTime: TimeSpan.Zero,
                MaxPendingRequests: Math.Max(8, cameras.Count * 4)),
            batchInputAssembler: new ConcatOrtValueRequestBatchInputAssembler(),
            outputShapeResolver: new YoloNmsOutputShapeResolver(defaultItemCount: 300, fieldCount: 7),
            outputDecoder: new CropTextObbOutputDecoder(ocr.ScoreThreshold));

        var recognitionEndpoint = new OrtValueBatchedInferenceEndpoint<PaddleOCRRecExtractor.OcrResult>(
            name: "multicamera.paddle.rec",
            modelPath: ocr.RecognitionModelPath!,
            executionBackend: ocr.RecognitionBackend,
            options: new OnnxBatchedResourceOptions(
                MaxBatchSize: 1,
                MaxWaitTime: TimeSpan.Zero,
                MaxPendingRequests: Math.Max(8, cameras.Count * 4)),
            batchInputAssembler: new ConcatOrtValueRequestBatchInputAssembler(),
            outputShapeResolver: new PaddleOCRRecOutputShapeResolver(ocr.RecognitionInputWidth),
            outputDecoder: new PaddleOCRRecOrtValueOutputDecoder());

        return new SharedInferenceEndpoints(boxDetectors, obbDetectors, textEndpoint, recognitionEndpoint);
    }

    public async ValueTask DisposeAsync()
    {
        foreach(OrtValueBatchedInferenceEndpoint<YoloBox> endpoint in boxDetectors.Values)
            await endpoint.DisposeAsync().ConfigureAwait(false);

        foreach(OrtValueBatchedInferenceEndpoint<YoloObb> endpoint in obbDetectors.Values)
            await endpoint.DisposeAsync().ConfigureAwait(false);

        if(TextDetector is not null)
            await TextDetector.DisposeAsync().ConfigureAwait(false);

        if(Recognition is not null)
            await Recognition.DisposeAsync().ConfigureAwait(false);
    }
}
