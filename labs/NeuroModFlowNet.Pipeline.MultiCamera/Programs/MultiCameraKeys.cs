namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Register names shared by the per-camera <c>prepare</c> program and the <c>common</c> program. The contract
/// between the two programs is exactly <see cref="ImageRect"/>: prepare must leave a U8 NHWC BGR tensor in model
/// device memory there; common reads it (ADR-001).
///
/// RU: Имена регистров, общие для prepare-программы камеры и common-программы. Контракт между программами это
/// ровно <see cref="ImageRect"/>: prepare обязан оставить там U8 NHWC BGR тензор в памяти устройства модели; common
/// его читает (ADR-001).
/// </summary>
internal static class MultiCameraKeys
{
    // Host -> prepare. Owner of the decoded frame, disposed with the run context.
    public const string SourceFrame = "image.source.frame";

    // CPU decode: BGR Mat.
    public const string SourceImage = "image.source";

    // NVDEC decode: zero-copy NV12 view over the CUDA surface, [allocHeight * 3 / 2, pitch] U8.
    public const string SourceNv12 = "image.source.nv12";

    public const string SourceTensor = "image.source.tensor";
    public const string SourceDevice = "image.source.device";

    // prepare -> common (contract).
    public const string ImageRect = "image.rect";

    // common: detector.
    public const string ImageRectRgb = "image.rect.rgb";
    public const string DetectionsRect = "det.rect";
    public const string TrackDetections = "track.det.in";

    // common: tracking.
    public const string Tracks = "track.out";
    public const string TrackerState = "tracker.state";
    public const string TrackerGate = "tracker";
    public const string TrackerStartZone = "tracker.startZone";
    public const string TrackerEndZone = "tracker.endZone";

    // common: bag crops -> text detection -> recognition.
    public const string CropBoxes = "ocr.crop.boxes";
    public const string CropTrackIds = "ocr.crop.trackIds";
    public const string CropBatch = "ocr.crop.batch";
    public const string CropTransforms = "ocr.crop.transforms";
    public const string CropCount = "ocr.crop.count";
    public const string TextCropRegions = "text.crop.regions";
    public const string TextObbRect = "text.obb.rect";
    public const string TextTrackIds = "text.trackIds";
    public const string OcrRecInput = "ocr.recInput";
    public const string OcrRoiCount = "ocr.roiCount";
    public const string OcrRecognition = "ocr.recognition";

    // common: preview.
    public const string PreviewTensor = "image.preview.tensor";
    public const string PreviewImage = "image.preview";

    public static string PrepareStage(int index) => $"image.prepare.{index}";

    public static string PrepareTransform(int index) => $"transform.prepare.{index}";

    // common: detector, one pass over the whole rectified strip.
    public const string DetectorResized = "det.resized";
    public const string DetectorResizeTransform = "det.resized.toRect";
    public const string DetectorModelInput = "det.model";
    public const string DetectorModelOutput = "det.out.model";
}
