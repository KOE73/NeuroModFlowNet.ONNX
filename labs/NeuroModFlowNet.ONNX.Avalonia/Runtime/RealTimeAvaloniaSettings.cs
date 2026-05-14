using System.Configuration;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Stores model, backend, and input-size settings for the Avalonia realtime lab.
/// RU: Хранит настройки моделей, backend-ов и входного размера для Avalonia realtime lab.
/// </summary>
/// <remarks>
/// EN: Values are read from <c>App.config</c> once at startup. This keeps configuration parsing out of the UI and the hot
/// inference loop.
/// RU: Значения читаются из <c>App.config</c> один раз при запуске. Так парсинг конфигурации не попадает в UI и hot
/// inference loop.
/// </remarks>
public sealed class RealTimeAvaloniaSettings
{
    const string InferenceBackendConfigKey = "InferenceBackend";
    const string ModelPrecisionConfigKey = "ModelPrecision";
    const string UseByteBgrConfigKey = "UseByteBgr";
    const string PaddleDetInferenceBackendConfigKey = "PaddleDetInferenceBackend";
    const string PaddleDetModelPrecisionConfigKey = "PaddleDetModelPrecision";
    const string PaddleDetUseByteBgrConfigKey = "PaddleDetUseByteBgr";
    const string PaddleRecInferenceBackendConfigKey = "PaddleRecInferenceBackend";
    const string PaddleRecModelPrecisionConfigKey = "PaddleRecModelPrecision";
    const string PaddleRecUseByteBgrConfigKey = "PaddleRecUseByteBgr";
    const string PaddleRecModelPathConfigKey = "PaddleRecModelPath";
    const string InputSizeConfigKey = "InputSize";
    const string BoxModelNameConfigKey = "BoxModelName";
    const string ObbModelNameConfigKey = "ObbModelName";
    const string SegModelNameConfigKey = "SegModelName";
    const string ClsModelNameConfigKey = "ClsModelName";
    const string PoseModelNameConfigKey = "PoseModelName";

    const InferenceBackend DefaultInferenceBackend = InferenceBackend.Cuda;
    const string DefaultModelPrecision = "fp16";
    const bool DefaultUseByteBgr = true;
    const InferenceBackend DefaultPaddleInferenceBackend = InferenceBackend.Cuda;
    const string DefaultPaddleModelPrecision = "fp32";
    const bool DefaultPaddleUseByteBgr = false;
    const int DefaultInputSize = 640;
    const string DefaultBoxModelName = "yolo26n";
    const string DefaultObbModelName = "img-text-to-obb";
    const string DefaultSegModelName = "yolo26n-seg";
    const string DefaultClsModelName = "yolo26n-cls";
    const string DefaultPoseModelName = "yolo26n-pose";

    public InferenceBackend InferenceBackend { get; init; } = DefaultInferenceBackend;
    public string ModelPrecision { get; init; } = DefaultModelPrecision;
    public bool UseByteBgr { get; init; } = DefaultUseByteBgr;
    public InferenceBackend PaddleDetInferenceBackend { get; init; } = DefaultPaddleInferenceBackend;
    public string PaddleDetModelPrecision { get; init; } = DefaultPaddleModelPrecision;
    public bool PaddleDetUseByteBgr { get; init; } = DefaultPaddleUseByteBgr;
    public InferenceBackend PaddleRecInferenceBackend { get; init; } = DefaultPaddleInferenceBackend;
    public string PaddleRecModelPrecision { get; init; } = DefaultPaddleModelPrecision;
    public bool PaddleRecUseByteBgr { get; init; } = DefaultPaddleUseByteBgr;
    public string? PaddleRecModelPath { get; init; }
    public int InputSize { get; init; } = DefaultInputSize;
    public string BoxModelName { get; init; } = DefaultBoxModelName;
    public string ObbModelName { get; init; } = DefaultObbModelName;
    public string SegModelName { get; init; } = DefaultSegModelName;
    public string ClsModelName { get; init; } = DefaultClsModelName;
    public string PoseModelName { get; init; } = DefaultPoseModelName;

    public static RealTimeAvaloniaSettings FromConfig() =>
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
            PaddleRecModelPath = ReadOptionalString(PaddleRecModelPathConfigKey),
            InputSize = ReadPositiveInt(InputSizeConfigKey, DefaultInputSize),
            BoxModelName = ReadString(BoxModelNameConfigKey, DefaultBoxModelName),
            ObbModelName = ReadString(ObbModelNameConfigKey, DefaultObbModelName),
            SegModelName = ReadString(SegModelNameConfigKey, DefaultSegModelName),
            ClsModelName = ReadString(ClsModelNameConfigKey, DefaultClsModelName),
            PoseModelName = ReadString(PoseModelNameConfigKey, DefaultPoseModelName),
        };

    static string ReadString(string key, string fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
    }

    static string? ReadOptionalString(string key)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    }

    static bool ReadBool(string key, bool fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return bool.TryParse(value, out bool result) ? result : fallback;
    }

    static int ReadPositiveInt(string key, int fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return int.TryParse(value, out int result) && result > 0 ? result : fallback;
    }

    static InferenceBackend ReadInferenceBackend(string key, InferenceBackend fallback)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return Enum.TryParse(value, ignoreCase: true, out InferenceBackend result) ? result : fallback;
    }
}
