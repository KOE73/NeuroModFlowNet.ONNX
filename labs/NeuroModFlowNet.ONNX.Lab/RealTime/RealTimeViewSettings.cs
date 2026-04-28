using System.Configuration;
using NeuroModFlowNet.ONNX;

namespace OnnxTestLoader;

public sealed class RealTimeViewSettings
{
    private const string InferenceBackendConfigKey = "InferenceBackend";
    private const string ModelPrecisionConfigKey = "ModelPrecision";
    private const string IsByteBgrConfigKey = "IsByteBgr";
    private const string InputSizeConfigKey = "InputSize";

    private const InferenceBackend DefaultInferenceBackend = InferenceBackend.TensorRt;
    private const string DefaultModelPrecision = "fp16";
    private const bool DefaultIsByteBgr = true;
    private const int DefaultInputSize = 640;

    public InferenceBackend InferenceBackend { get; init; } = DefaultInferenceBackend;
    public string ModelPrecision { get; init; } = DefaultModelPrecision;
    public bool IsByteBgr { get; init; } = DefaultIsByteBgr;
    public int InputSize { get; init; } = DefaultInputSize;

    public static RealTimeViewSettings FromConfig() =>
        new()
        {
            InferenceBackend = ReadInferenceBackend(InferenceBackendConfigKey, DefaultInferenceBackend),
            ModelPrecision = ReadString(ModelPrecisionConfigKey, DefaultModelPrecision),
            IsByteBgr = ReadBool(IsByteBgrConfigKey, DefaultIsByteBgr),
            InputSize = ReadPositiveInt(InputSizeConfigKey, DefaultInputSize),
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
