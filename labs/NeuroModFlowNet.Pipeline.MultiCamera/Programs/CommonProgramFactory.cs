using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;
using CvSize = OpenCvSharp.Size;
using CvRect = OpenCvSharp.Rect;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Builds the <c>common</c> program that is identical as code for every camera: rectified U8 NHWC frame ->
/// resize to the camera's detector input (RGB NCHW) -> object detection -> coordinates back to rect space ->
/// tracker (ordered critical section, state in GlobalMemory) -> crops of confirmed
/// tracks (GPU letterbox batch) -> text OBB on the crops -> text back to rect space with track ids -> PaddleOCR
/// recognition on the full-resolution rect frame -> optional CPU preview. Each controller receives its own instance;
/// endpoints are shared.
///
/// RU: Собирает common-программу, одинаковую как код для всех камер: выпрямленный U8 NHWC кадр ->
/// изменение размера под вход детектора камеры (RGB NCHW) -> детекция объектов -> координаты обратно в rect-пространство ->
/// трекер (ordered critical section, state в GlobalMemory) -> вырезки
/// подтверждённых треков (GPU letterbox-батч) -> text OBB на вырезках -> текст обратно в rect с id треков -> PaddleOCR
/// на полноразмерном rect-кадре -> опционально CPU-превью. У каждого контроллера свой экземпляр; endpoint-ы общие.
/// </summary>
internal static class CommonProgramFactory
{
    public static VmProgram Create(
        MultiCameraConfig config,
        CameraConfig camera,
        CameraGeometry geometry,
        SharedInferenceEndpoints endpoints)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(camera);
        ArgumentNullException.ThrowIfNull(endpoints);

        InferenceBackend backend = config.Backend;
        var builder = new VmProgramBuilder("common");

        if(config.Ocr.Enabled)
        {
            // Use the full-resolution RGB FP32 rect frame for track crops and OCR region extraction.
            builder.Step(new Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
                MultiCameraKeys.ImageRect,
                MultiCameraKeys.ImageRectRgb,
                isFinal: false,
                executionBackend: backend));
        }

        AppendDetector(builder, config.Detector, camera, geometry, backend, endpoints);

        TrackerConfig tracker = config.Tracker;
        builder.Step(new Op_Track(
            MultiCameraKeys.TrackDetections,
            MultiCameraKeys.Tracks,
            MultiCameraKeys.TrackerState,
            MultiCameraKeys.TrackerGate,
            new IouTrackerOptions(
                tracker.IouThreshold,
                tracker.MinAge,
                tracker.MaxMissedFrames,
                tracker.MaxTrailLength,
                tracker.StartZoneContainmentThreshold),
            startZoneKey: camera.Tracking?.StartZone is null ? null : MultiCameraKeys.TrackerStartZone,
            endZoneKey: camera.Tracking?.EndZone is null ? null : MultiCameraKeys.TrackerEndZone));

        if(config.Ocr.Enabled)
            AppendOcr(builder, config.Ocr, backend, endpoints);

        if(config.Preview.Enabled)
            AppendPreview(builder, config.Preview, geometry, backend);

        return builder.Build();
    }

    #region Detector

    /// <summary>
    /// Resize the rectified frame to the model input (stretch or letterbox), convert to RGB NCHW, detect, then map
    /// boxes back to rect space. OCR extracts regions from the full-resolution rect frame to preserve text detail.
    /// </summary>
    static void AppendDetector(
        VmProgramBuilder builder,
        DetectorConfig detector,
        CameraConfig camera,
        CameraGeometry geometry,
        InferenceBackend backend,
        SharedInferenceEndpoints endpoints)
    {
        CameraDetectorConfig model = camera.Detector;
        var inputSize = new CvSize(model.InputWidth, model.InputHeight);

        if(model.CropX is int cropX && model.CropY is int cropY)
        {
            if(cropX > geometry.Width - inputSize.Width || cropY > geometry.Height - inputSize.Height)
                throw new InvalidDataException($"camera '{camera.Id}': detector crop does not fit the prepared {geometry.Width}x{geometry.Height} frame.");
            builder.Step(new Op_Onnx_Crop_U8_NHWC(
                MultiCameraKeys.ImageRect,
                MultiCameraKeys.DetectorResized,
                new CvRect(cropX, cropY, inputSize.Width, inputSize.Height),
                outputTransformKey: MultiCameraKeys.DetectorResizeTransform,
                isFinal: false,
                executionBackend: model.CropBackend ?? backend));
        }
        else if(detector.IsStretch)
        {
            builder.Step(new Op_Onnx_Resize_U8_NHWC(
                MultiCameraKeys.ImageRect,
                MultiCameraKeys.DetectorResized,
                inputSize,
                outputTransformKey: MultiCameraKeys.DetectorResizeTransform,
                isFinal: false,
                executionBackend: backend));
        }
        else
        {
            builder.Step(new Op_Onnx_PadResize_U8_NHWC(
                MultiCameraKeys.ImageRect,
                MultiCameraKeys.DetectorResized,
                inputSize,
                outputTransformKey: MultiCameraKeys.DetectorResizeTransform,
                stride: 32,
                mode: PadResizeMode.FixedCanvas,
                isFinal: false,
                executionBackend: backend));
        }

        builder.Step(detector.IsFp16(model.ModelPath)
            ? new Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255(MultiCameraKeys.DetectorResized, MultiCameraKeys.DetectorModelInput, isFinal: false, executionBackend: backend)
            : new Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(MultiCameraKeys.DetectorResized, MultiCameraKeys.DetectorModelInput, isFinal: false, executionBackend: backend));

        if(detector.IsBox)
        {
            builder.Step(new Model_OrtValueInference<YoloBox>(
                name: "Model_MultiCamera_Detector_Box",
                inputKey: MultiCameraKeys.DetectorModelInput,
                outputKey: MultiCameraKeys.DetectorModelOutput,
                endpoint: endpoints.GetBoxDetector(model.ModelPath)));
        }
        else
        {
            builder.Step(new Model_OrtValueInference<YoloObb>(
                name: "Model_MultiCamera_Detector_OBB",
                inputKey: MultiCameraKeys.DetectorModelInput,
                outputKey: MultiCameraKeys.DetectorModelOutput,
                endpoint: endpoints.GetObbDetector(model.ModelPath)));
        }

        // Resize / pad-resize are scale + offset: box corners map back exactly.
        builder.Step(new Op_Map_Coordinates(
            MultiCameraKeys.DetectorModelOutput,
            MultiCameraKeys.DetectorResizeTransform,
            MultiCameraKeys.DetectionsRect,
            CoordinateMappingShapePolicy.PreserveShape));

        builder.Step(detector.IsBox
            ? new Op_DetectionsFromYoloBoxArray(MultiCameraKeys.DetectionsRect, MultiCameraKeys.TrackDetections, detector.TrackedClassId)
            : new Op_DetectionsFromYoloObbArray(MultiCameraKeys.DetectionsRect, MultiCameraKeys.TrackDetections, detector.TrackedClassId));
    }

    #endregion

    #region OCR

    /// <summary>
    /// Crop confirmed tracks, detect text on the crops, then recognize regions taken from the full-resolution rect
    /// frame. The text stages are skipped when the preceding step finds no crops or text regions.
    /// </summary>
    static void AppendOcr(VmProgramBuilder builder, OcrConfig ocr, InferenceBackend backend, SharedInferenceEndpoints endpoints)
    {
        builder.Step(new Op_TracksToCropBoxes(
            MultiCameraKeys.Tracks,
            MultiCameraKeys.CropBoxes,
            MultiCameraKeys.CropTrackIds,
            ocr.MaxCrops));

        static bool HasCrops(VmRunContext context) =>
            context.TryGet(MultiCameraKeys.CropBoxes, out YoloObb[] crops) && crops.Length > 0;

        static bool HasTextRegions(VmRunContext context) =>
            context.TryGet(MultiCameraKeys.TextObbRect, out YoloObb[] regions) && regions.Length > 0;

        builder.Label("ocr.crops").Step(new OpConditional(
            new Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW(
                imageInputKey: MultiCameraKeys.ImageRectRgb,
                obbInputKey: MultiCameraKeys.CropBoxes,
                outputKey: MultiCameraKeys.CropBatch,
                targetSize: new CvSize(ocr.DetectionInputSize, ocr.DetectionInputSize),
                maxRoiCount: ocr.MaxCrops,
                outputTransformsKey: MultiCameraKeys.CropTransforms,
                actualCountOutputKey: MultiCameraKeys.CropCount,
                paddingPixels: ocr.CropPaddingPixels,
                paddingScale: ocr.CropPaddingScale,
                overflowPolicy: PaddleRecRoiOverflowPolicy.Truncate,
                isFinal: false,
                executionBackend: backend),
            HasCrops, "ocr.hasCrops"));

        builder.Step(new OpConditional(
            new Model_OrtValueInference<CropTextRegion>(
                name: "Model_MultiCamera_TextObb_OnCrops",
                inputKey: MultiCameraKeys.CropBatch,
                outputKey: MultiCameraKeys.TextCropRegions,
                endpoint: endpoints.TextDetector!),
            HasCrops, "ocr.hasCrops"));

        builder.Step(new OpConditional(
            new Op_MapCropTextRegionsToRect(
                MultiCameraKeys.TextCropRegions,
                MultiCameraKeys.CropTransforms,
                MultiCameraKeys.CropCount,
                MultiCameraKeys.CropTrackIds,
                MultiCameraKeys.TextObbRect,
                MultiCameraKeys.TextTrackIds),
            HasCrops, "ocr.hasCrops"));

        builder.Label("ocr.recognition").Step(new OpConditional(
            new Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW(
                imageInputKey: MultiCameraKeys.ImageRectRgb,
                obbInputKey: MultiCameraKeys.TextObbRect,
                outputKey: MultiCameraKeys.OcrRecInput,
                targetSize: new CvSize(ocr.RecognitionInputWidth, ocr.RecognitionInputHeight),
                paddingPixels: ocr.PaddingPixels,
                paddingScale: ocr.PaddingScale,
                maxRoiCount: ocr.MaxRoiCount,
                overflowPolicy: ocr.MaxRoiCount is null ? PaddleRecRoiOverflowPolicy.Fail : PaddleRecRoiOverflowPolicy.Truncate,
                actualCountOutputKey: MultiCameraKeys.OcrRoiCount,
                isFinal: false,
                executionBackend: backend),
            HasTextRegions, "ocr.hasTextRegions"));

        builder.Step(new OpConditional(
            new Model_OrtValueInference<PaddleOCRRecExtractor.OcrResult>(
                name: "Model_MultiCamera_PaddleRec",
                inputKey: MultiCameraKeys.OcrRecInput,
                outputKey: MultiCameraKeys.OcrRecognition,
                endpoint: endpoints.Recognition!),
            HasTextRegions, "ocr.hasTextRegions"));
    }

    #endregion

    static void AppendPreview(VmProgramBuilder builder, PreviewConfig preview, CameraGeometry geometry, InferenceBackend backend)
    {
        double scale = Math.Min(1.0, preview.MaxEdge / (double)Math.Max(geometry.Width, geometry.Height));
        var previewSize = new CvSize(
            Math.Max(1, (int)Math.Round(geometry.Width * scale)),
            Math.Max(1, (int)Math.Round(geometry.Height * scale)));

        // isFinal: true places the resized tensor in CPU memory so it can be copied into a Mat.
        builder.Step(new Op_Onnx_Resize_U8_NHWC(
            MultiCameraKeys.ImageRect,
            MultiCameraKeys.PreviewTensor,
            previewSize,
            isFinal: true,
            executionBackend: backend));

        builder.Step(new Copy_OrtValue_To_MatImage(MultiCameraKeys.PreviewTensor, MultiCameraKeys.PreviewImage));
    }
}
