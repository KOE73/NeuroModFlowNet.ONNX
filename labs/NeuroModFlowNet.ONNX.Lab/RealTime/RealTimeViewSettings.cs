using System.Configuration;
using NeuroModFlowNet.ONNX;

namespace OnnxTestLoader;

public sealed class RealTimeViewSettings
{
    private const string InferenceBackendConfigKey = "InferenceBackend";
    private const string ModelPrecisionConfigKey = "ModelPrecision";
    private const string UseByteBgrConfigKey = "UseByteBgr";
    private const string PaddleDetInferenceBackendConfigKey = "PaddleDetInferenceBackend";
    private const string PaddleDetModelPrecisionConfigKey = "PaddleDetModelPrecision";
    private const string PaddleDetUseByteBgrConfigKey = "PaddleDetUseByteBgr";
    private const string PaddleRecInferenceBackendConfigKey = "PaddleRecInferenceBackend";
    private const string PaddleRecModelPrecisionConfigKey = "PaddleRecModelPrecision";
    private const string PaddleRecUseByteBgrConfigKey = "PaddleRecUseByteBgr";
    private const string InputSizeConfigKey = "InputSize";
    private const string BoxModelNameConfigKey = "BoxModelName";
    private const string ObbModelNameConfigKey = "ObbModelName";
    private const string PoseModelNameConfigKey = "PoseModelName";
    private const string SegModelNameConfigKey = "SegModelName";

    private const InferenceBackend DefaultInferenceBackend = InferenceBackend.TensorRt;
    private const string DefaultModelPrecision = "fp16";
    private const bool DefaultUseByteBgr = true;
    private const InferenceBackend DefaultPaddleInferenceBackend = InferenceBackend.Cuda;
    private const string DefaultPaddleModelPrecision = "fp32";
    private const bool DefaultPaddleUseByteBgr = false;
    private const int DefaultInputSize = 640;
    private const string DefaultBoxModelName = "yolo26s";
    private const string DefaultObbModelName = "yolo26s-obb";
    private const string DefaultPoseModelName = "yolo26s-pose";
    private const string DefaultSegModelName = "yolo26s-seg";

    public InferenceBackend InferenceBackend { get; init; } = DefaultInferenceBackend;
    public string ModelPrecision { get; init; } = DefaultModelPrecision;
    public bool UseByteBgr { get; init; } = DefaultUseByteBgr;
    public InferenceBackend PaddleDetInferenceBackend { get; init; } = DefaultPaddleInferenceBackend;
    public string PaddleDetModelPrecision { get; init; } = DefaultPaddleModelPrecision;
    public bool PaddleDetUseByteBgr { get; init; } = DefaultPaddleUseByteBgr;
    public InferenceBackend PaddleRecInferenceBackend { get; init; } = DefaultPaddleInferenceBackend;
    public string PaddleRecModelPrecision { get; init; } = DefaultPaddleModelPrecision;
    public bool PaddleRecUseByteBgr { get; init; } = DefaultPaddleUseByteBgr;
    public int InputSize { get; init; } = DefaultInputSize;
    public string BoxModelName { get; init; } = DefaultBoxModelName;
    public string ObbModelName { get; init; } = DefaultObbModelName;
    public string PoseModelName { get; init; } = DefaultPoseModelName;
    public string SegModelName { get; init; } = DefaultSegModelName;

    public static RealTimeViewSettings FromConfig() =>
        new()
        {
            InferenceBackend = ReadInferenceBackend(InferenceBackendConfigKey, DefaultInferenceBackend),
            ModelPrecision = ReadString(ModelPrecisionConfigKey, DefaultModelPrecision),
            UseByteBgr = ReadBool(UseByteBgrConfigKey, DefaultUseByteBgr),
            PaddleDetInferenceBackend = ReadInferenceBackend(PaddleDetInferenceBackendConfigKey, DefaultPaddleInferenceBackend),
            PaddleDetModelPrecision = ReadString(PaddleDetModelPrecisionConfigKey, DefaultPaddleModelPrecision),
            PaddleDetUseByteBgr = ReadBool(PaddleDetUseByteBgrConfigKey, DefaultPaddleUseByteBgr),
            PaddleRecInferenceBackend = ReadInferenceBackend(PaddleRecInferenceBackendConfigKey, DefaultPaddleInferenceBackend),
            PaddleRecModelPrecision = ReadString(PaddleRecModelPrecisionConfigKey, DefaultPaddleModelPrecision),
            PaddleRecUseByteBgr = ReadBool(PaddleRecUseByteBgrConfigKey, DefaultPaddleUseByteBgr),
            InputSize = ReadPositiveInt(InputSizeConfigKey, DefaultInputSize),
            BoxModelName = ReadString(BoxModelNameConfigKey, DefaultBoxModelName),
            ObbModelName = ReadString(ObbModelNameConfigKey, DefaultObbModelName),
            PoseModelName = ReadString(PoseModelNameConfigKey, DefaultPoseModelName),
            SegModelName = ReadString(SegModelNameConfigKey, DefaultSegModelName),
        };

    private static string ReadString(string key, string fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
    }

    private static bool ReadBool(string key, bool fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return bool.TryParse(value, out bool result) ? result : fallback;
    }

    private static int ReadPositiveInt(string key, int fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return int.TryParse(value, out int result) && result > 0 ? result : fallback;
    }

    private static InferenceBackend ReadInferenceBackend(string key, InferenceBackend fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return Enum.TryParse(value, ignoreCase: true, out InferenceBackend result) ? result : fallback;
    }
}
