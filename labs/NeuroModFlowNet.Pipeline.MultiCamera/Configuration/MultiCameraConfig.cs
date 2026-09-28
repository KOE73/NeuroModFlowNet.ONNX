using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Root of the multi-camera lab configuration (JSON). Shared model/tracker settings plus one entry per camera.
///
/// RU: Корень конфигурации лаборатории (JSON). Общие настройки моделей/трекера плюс запись на каждую камеру.
/// </summary>
internal sealed record MultiCameraConfig(
    InferenceBackend Backend,
    DetectorConfig Detector,
    TrackerConfig Tracker,
    OcrConfig Ocr,
    PreviewConfig Preview,
    IReadOnlyList<CameraConfig> Cameras)
{
    public IEnumerable<CameraConfig> EnabledCameras => Cameras.Where(static camera => camera.Enabled);

    public void Validate()
    {
        ArgumentNullException.ThrowIfNull(Detector);
        ArgumentNullException.ThrowIfNull(Tracker);
        ArgumentNullException.ThrowIfNull(Ocr);
        ArgumentNullException.ThrowIfNull(Preview);
        Detector.Validate();
        Ocr.Validate();
        Preview.Validate();

        if(Cameras is null || !EnabledCameras.Any())
            throw new InvalidDataException("At least one enabled camera is required.");

        var seenIds = new HashSet<string>(StringComparer.Ordinal);
        foreach(CameraConfig camera in Cameras)
        {
            camera.Validate();
            if(!seenIds.Add(camera.Id))
                throw new InvalidDataException($"Duplicate camera id '{camera.Id}'.");
        }
    }
}

/// <summary>
/// EN: Detector settings shared by all cameras. <c>Kind</c> "box" = YOLO box model with NMS inside (output [1,300,6]),
/// "obb" = OBB model (output [1,300,7]). <c>InputMode</c> "stretch" resizes the whole rectified strip to the camera's
/// model input (<see cref="CameraDetectorConfig"/>), "letterbox" pads it. The model file and its input size are per
/// camera, because each strip has its own size and the input is chosen so that a bag has the pixel size seen in training.
///
/// RU: Общие для всех камер настройки детектора. <c>Kind</c> "box" = YOLO box модель с NMS внутри (выход [1,300,6]),
/// "obb" = OBB модель (выход [1,300,7]). <c>InputMode</c> "stretch" растягивает всю выпрямленную полосу на вход модели
/// камеры (<see cref="CameraDetectorConfig"/>), "letterbox" добавляет поля. Файл модели и размер входа задаются у
/// камеры: у каждой полосы свой размер, а вход подобран так, чтобы мешок был того же размера в пикселях, что при обучении.
/// </summary>
internal sealed record DetectorConfig(
    string Kind = "box",
    InferenceBackend? Backend = null,
    string? Precision = null,
    string InputMode = "stretch",
    float ScoreThreshold = 0.5f,
    int MaxItemCount = 300,
    int? TrackedClassId = null)
{
    public bool IsBox => string.Equals(Kind, "box", StringComparison.OrdinalIgnoreCase);

    public bool IsStretch => string.Equals(InputMode, "stretch", StringComparison.OrdinalIgnoreCase);

    public void Validate()
    {
        if(!IsBox && !string.Equals(Kind, "obb", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("detector.kind must be \"box\" or \"obb\".");
        if(Precision is not null && !string.Equals(Precision, "fp16", StringComparison.OrdinalIgnoreCase) && !string.Equals(Precision, "fp32", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("detector.precision must be \"fp16\", \"fp32\" or omitted.");
        if(!IsStretch && !string.Equals(InputMode, "letterbox", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("detector.inputMode must be \"stretch\" or \"letterbox\".");
        if(ScoreThreshold is < 0 or > 1)
            throw new InvalidDataException("detector.scoreThreshold must be in [0, 1].");
        if(MaxItemCount <= 0)
            throw new InvalidDataException("detector.maxItemCount must be positive.");
    }

    /// <summary>FP16 input when set explicitly, otherwise from the file name ("fp16" or "half").</summary>
    public bool IsFp16(string modelPath) => Precision is null
        ? Path.GetFileNameWithoutExtension(modelPath).Contains("fp16", StringComparison.OrdinalIgnoreCase) ||
          Path.GetFileNameWithoutExtension(modelPath).Contains("half", StringComparison.OrdinalIgnoreCase)
        : string.Equals(Precision, "fp16", StringComparison.OrdinalIgnoreCase);
}

/// <summary>
/// EN: Per-camera detector model: a static model exported for this camera's strip. <c>InputWidth</c> x <c>InputHeight</c>
/// must equal the model input exactly (checked on the first run by the endpoint).
///
/// RU: Модель детектора камеры: статическая модель, выгруженная под полосу этой камеры. <c>InputWidth</c> x
/// <c>InputHeight</c> обязаны точно совпадать с входом модели (проверяется endpoint-ом на первом запуске).
/// </summary>
internal sealed record CameraDetectorConfig(string ModelPath, int InputWidth, int InputHeight)
{
    public void Validate(string cameraId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath, $"camera '{cameraId}': detector.modelPath");
        if(InputWidth <= 0 || InputHeight <= 0 || InputWidth % 32 != 0 || InputHeight % 32 != 0)
            throw new InvalidDataException($"camera '{cameraId}': detector input must be positive multiples of 32, got {InputWidth}x{InputHeight}.");
    }
}

internal sealed record TrackerConfig(
    float IouThreshold = 0.5f,
    int MinAge = 5,
    int MaxMissedFrames = 25,
    int MaxTrailLength = 50,
    float StartZoneContainmentThreshold = 1f);

/// <summary>
/// EN: OCR on bag crops. <c>MaxCrops</c> is the fixed crop batch (must equal the text model's fixed batch, for example 4
/// for <c>img-text-to-obb__640_b4_fp32.onnx</c>); <c>DetectionInputSize</c> is the crop canvas; crop padding widens the
/// track box before cutting. The text model must be FP32: the crop batch is produced in FP32.
///
/// RU: OCR на вырезках мешков. <c>MaxCrops</c> это фиксированный батч вырезок (должен равняться фиксированному батчу
/// текстовой модели, например 4 для <c>img-text-to-obb__640_b4_fp32.onnx</c>); <c>DetectionInputSize</c> это холст
/// вырезки; отступ расширяет бокс трека перед вырезкой. Текстовая модель должна быть FP32: батч вырезок делается в FP32.
/// </summary>
internal sealed record OcrConfig(
    bool Enabled,
    string? TextObbModelPath,
    string? RecognitionModelPath,
    InferenceBackend RecognitionBackend = InferenceBackend.TensorRt,
    int DetectionInputSize = 640,
    int MaxCrops = 4,
    float CropPaddingPixels = 12f,
    float CropPaddingScale = 0.05f,
    float ScoreThreshold = 0.25f,
    int RecognitionInputWidth = 640,
    int RecognitionInputHeight = 48,
    float PaddingPixels = 2f,
    float PaddingScale = 0.10f,
    int? MaxRoiCount = 16)
{
    public void Validate()
    {
        if(!Enabled)
            return;

        ArgumentException.ThrowIfNullOrWhiteSpace(TextObbModelPath, "ocr.textObbModelPath");
        ArgumentException.ThrowIfNullOrWhiteSpace(RecognitionModelPath, "ocr.recognitionModelPath");
        if(DetectionInputSize <= 0 || DetectionInputSize % 32 != 0)
            throw new InvalidDataException("ocr.detectionInputSize must be a positive multiple of 32.");
        if(MaxCrops <= 0)
            throw new InvalidDataException("ocr.maxCrops must be positive.");
        if(CropPaddingPixels < 0 || CropPaddingScale < 0)
            throw new InvalidDataException("ocr crop padding must be non-negative.");
        if(Path.GetFileNameWithoutExtension(TextObbModelPath).Contains("fp16", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("ocr.textObbModelPath must be an FP32 model: the crop batch is produced in FP32.");
        if(RecognitionInputWidth <= 0 || RecognitionInputHeight <= 0)
            throw new InvalidDataException("ocr recognition input size must be positive.");
        if(MaxRoiCount is <= 0)
            throw new InvalidDataException("ocr.maxRoiCount must be positive or null.");
    }
}

internal sealed record PreviewConfig(bool Enabled = true, int MaxEdge = 900)
{
    public void Validate()
    {
        if(Enabled && MaxEdge < 64)
            throw new InvalidDataException("preview.maxEdge must be at least 64.");
    }
}

internal sealed record CameraConfig(
    string Id,
    SourceConfig Source,
    SizeConfig Resolution,
    IReadOnlyList<TransformConfig> Transforms,
    TrackingZonesConfig? Tracking,
    CameraDetectorConfig Detector,
    bool Enabled = true,
    int MaxInFlight = 1)
{
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(Id);
        ArgumentNullException.ThrowIfNull(Source, $"camera '{Id}': source");
        ArgumentNullException.ThrowIfNull(Resolution, $"camera '{Id}': resolution");
        Source.Validate(Id);
        if(Resolution.Width <= 0 || Resolution.Height <= 0)
            throw new InvalidDataException($"camera '{Id}': resolution must be positive.");
        if(MaxInFlight <= 0)
            throw new InvalidDataException($"camera '{Id}': maxInFlight must be positive.");

        foreach(TransformConfig transform in Transforms ?? [])
            transform.Validate(Id);

        ArgumentNullException.ThrowIfNull(Detector, $"camera '{Id}': detector");
        Detector.Validate(Id);
    }
}


internal sealed record SourceConfig(
    string Kind,
    string? Path = null,
    string? Pattern = null,
    IReadOnlyList<string>? Files = null,
    string? Url = null,
    int? Index = null,
    bool Loop = true,
    bool Realtime = true,
    string Transport = "tcp",
    string HwAcceleration = "none",
    string Decoder = "cpu",
    NeuroModFlowNet.Pipeline.Video.Nvdec.Nv12ColorMatrix ColorMatrix = NeuroModFlowNet.Pipeline.Video.Nvdec.Nv12ColorMatrix.Bt601Full)
{
    /// <summary>"nvdec": FFmpeg 9 NVDEC decode into CUDA memory; "cpu": OpenCV/FFmpeg decode into a Mat.</summary>
    public bool IsNvdec => string.Equals(Decoder, "nvdec", StringComparison.OrdinalIgnoreCase);

    public void Validate(string cameraId)
    {
        if(!IsNvdec && !string.Equals(Decoder, "cpu", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"camera '{cameraId}': source.decoder must be \"cpu\" or \"nvdec\".");
        if(IsNvdec && !string.Equals(HwAcceleration, "none", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"camera '{cameraId}': source.hwAcceleration applies to decoder \"cpu\" only.");

        switch(Kind?.ToLowerInvariant())
        {
            case "folder":
            case "file":
                ArgumentException.ThrowIfNullOrWhiteSpace(Path, $"camera '{cameraId}': source.path");
                break;
            case "files":
                if(Files is not { Count: > 0 } || Files.Any(string.IsNullOrWhiteSpace))
                    throw new InvalidDataException($"camera '{cameraId}': source.files must list at least one video file for kind 'files'.");
                break;
            case "rtsp":
                ArgumentException.ThrowIfNullOrWhiteSpace(Url, $"camera '{cameraId}': source.url");
                break;
            case "camera":
                if(Index is null or < 0)
                    throw new InvalidDataException($"camera '{cameraId}': source.index is required for kind 'camera'.");
                break;
            default:
                throw new InvalidDataException($"camera '{cameraId}': unknown source kind '{Kind}'.");
        }
    }
}

/// <summary>
/// EN: One geometric transform of the per-camera prepare program. Flat record: the fields used depend on <see cref="Type"/>
/// (radialUndistort, perspective, rotate, crop, resize), mirroring the CGPCam2026 cameras.json schema.
///
/// RU: Один геометрический шаг prepare-программы камеры. Плоская запись: набор полей зависит от <see cref="Type"/>.
/// </summary>
internal sealed record TransformConfig(
    string Type,
    float FocalLengthFactor = 0.85f,
    DistortionCoefficientsConfig? DistortionCoefficients = null,
    float Alpha = 0f,
    IReadOnlyList<PointConfig>? SourcePoints = null,
    SizeConfig? OutputSize = null,
    int Degrees = 90,
    int X = 0,
    int Y = 0,
    int Width = 0,
    int Height = 0)
{
    public void Validate(string cameraId)
    {
        switch(Type?.ToLowerInvariant())
        {
            case "radialundistort":
                if(DistortionCoefficients is null)
                    throw new InvalidDataException($"camera '{cameraId}': radialUndistort requires distortionCoefficients.");
                if(FocalLengthFactor <= 0)
                    throw new InvalidDataException($"camera '{cameraId}': focalLengthFactor must be positive.");
                if(Alpha is < 0 or > 1)
                    throw new InvalidDataException($"camera '{cameraId}': alpha must be in [0, 1].");
                break;
            case "perspective":
                if(SourcePoints is not { Count: 4 })
                    throw new InvalidDataException($"camera '{cameraId}': perspective requires exactly 4 sourcePoints (TL, TR, BR, BL).");
                if(OutputSize is null || OutputSize.Width <= 0 || OutputSize.Height <= 0)
                    throw new InvalidDataException($"camera '{cameraId}': perspective requires a positive outputSize.");
                break;
            case "rotate":
                if(Degrees is not (90 or 180 or 270))
                    throw new InvalidDataException($"camera '{cameraId}': rotate.degrees must be 90, 180 or 270.");
                break;
            case "crop":
                if(Width <= 0 || Height <= 0 || X < 0 || Y < 0)
                    throw new InvalidDataException($"camera '{cameraId}': crop requires non-negative x/y and positive width/height.");
                break;
            case "resize":
                if(Width <= 0 || Height <= 0)
                    throw new InvalidDataException($"camera '{cameraId}': resize requires positive width/height.");
                break;
            default:
                throw new InvalidDataException($"camera '{cameraId}': unknown transform type '{Type}'.");
        }
    }
}

internal sealed record DistortionCoefficientsConfig(float K1, float K2 = 0f, float P1 = 0f, float P2 = 0f, float K3 = 0f);

internal sealed record PointConfig(float X, float Y);

internal sealed record SizeConfig(int Width, int Height);

internal sealed record TrackingZonesConfig(ZoneConfig? StartZone, ZoneConfig? EndZone);

internal sealed record ZoneConfig(float X1, float Y1, float X2, float Y2);
