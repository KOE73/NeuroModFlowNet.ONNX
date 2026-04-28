using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// TODO Maked Gemini need check.
/// Конфигурация CUDA Execution Provider.
/// Оптимизировано для максимальной производительности на RTX 5090 (Blackwell).
/// </summary>
public sealed class CudaConfig : ExecutionProviderConfig
{
    public override InferenceBackend InferenceBackend => InferenceBackend.Cuda;

    // --- УПРАВЛЕНИЕ ПАМЯТЬЮ ---
    public double? GpuMemLimitGb { get; set; }
    /// <summary>
    /// Стратегия расширения арены: "kNextPowerOfTwo" (0) или "kSameAsRequested" (1).
    /// </summary>
    public string? ArenaExtendStrategy { get; set; }

    // --- АЛГОРИТМЫ И CUDNN ---
    /// <summary>
    /// Поиск алгоритмов свертки: "DEFAULT", "HEURISTIC", "EXHAUSTIVE".
    /// На 5090 EXHAUSTIVE может дать профит, но замедлит первый запуск.
    /// </summary>
    public string? CudnnConvAlgoSearch { get; set; }
    public bool? CudnnConvUseMaxWorkspace { get; set; }
    public bool? CudnnConvPadToNc1h1 { get; set; }
    public bool? DoCopyInDefaultStream { get; set; }

    // --- ПРОДВИНУТЫЕ ОПТИМИЗАЦИИ (CUDA GRAPH & TUNING) ---
    /// <summary>
    /// Включает CUDA Graphs для уменьшения оверхеда на запуск ядер. 
    /// Критично для маленьких моделей и Batch=1.
    /// </summary>
    public bool? EnableCudaGraph { get; set; }

    /// <summary>
    /// Tunable Op позволяет ORT подбирать лучшие параметры ядер на лету.
    /// </summary>
    public bool? TunableOpEnable { get; set; }
    public bool? TunableOpTuningEnable { get; set; }
    public int? TunableOpMaxTuningDurationMs { get; set; }

    public static CudaConfig FromDefaultOptions(OrtCUDAProviderOptions cudaOptions) =>
        FromDefaultOptions(cudaOptions.GetOptions());

    /// <summary>
    /// Создает конфиг из сырой строки параметров.
    /// </summary>
    public static CudaConfig FromDefaultOptions(string rawOptions)
    {
        var config = new CudaConfig();
        if(string.IsNullOrWhiteSpace(rawOptions)) return config;

        var dict = rawOptions.Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(part => part.Split('=', 2))
            .Where(part => part.Length == 2)
            .ToDictionary(sp => sp[0].Trim(), sp => sp[1].Trim());

        bool? ParseBool(string key) => dict.TryGetValue(key, out var v) ? (v == "1") : null;
        int? ParseInt(string key) => dict.TryGetValue(key, out var v) && int.TryParse(v, out var res) ? res : null;
        string? GetStr(string key) => dict.TryGetValue(key, out var v) ? v : null;

        config.DeviceId = ParseInt("device_id") ?? 0;

        if(dict.TryGetValue("gpu_mem_limit", out var memStr) && long.TryParse(memStr, out var memBytes))
            config.GpuMemLimitGb = memBytes / (1024.0 * 1024.0 * 1024.0);

        config.ArenaExtendStrategy = GetStr("arena_extend_strategy");
        config.CudnnConvAlgoSearch = GetStr("cudnn_conv_algo_search");
        config.CudnnConvUseMaxWorkspace = ParseBool("cudnn_conv_use_max_workspace");
        config.CudnnConvPadToNc1h1 = ParseBool("cudnn_conv_pad_to_nc1h1");
        config.DoCopyInDefaultStream = ParseBool("do_copy_in_default_stream");

        config.EnableCudaGraph = ParseBool("enable_cuda_graph");
        config.TunableOpEnable = ParseBool("tunable_op_enable");
        config.TunableOpTuningEnable = ParseBool("tunable_op_tuning_enable");
        config.TunableOpMaxTuningDurationMs = ParseInt("tunable_op_max_tuning_duration_ms");

        return config;
    }

    public override Dictionary<string, string> ToDictionary()
    {
        var dict = new Dictionary<string, string>();

        void AddIfNotNull(string key, object? value)
        {
            if(value is null) return;
            string stringValue = value is bool b ? (b ? "1" : "0") : value.ToString()!;
            if(!string.IsNullOrWhiteSpace(stringValue)) dict[key] = stringValue;
        }

        AddIfNotNull("device_id", DeviceId);

        if(GpuMemLimitGb.HasValue)
            dict["gpu_mem_limit"] = ((long)(GpuMemLimitGb.Value * 1024 * 1024 * 1024)).ToString();

        AddIfNotNull("arena_extend_strategy", ArenaExtendStrategy);
        AddIfNotNull("cudnn_conv_algo_search", CudnnConvAlgoSearch);
        AddIfNotNull("cudnn_conv_use_max_workspace", CudnnConvUseMaxWorkspace);
        AddIfNotNull("cudnn_conv_pad_to_nc1h1", CudnnConvPadToNc1h1);
        AddIfNotNull("do_copy_in_default_stream", DoCopyInDefaultStream);

        AddIfNotNull("enable_cuda_graph", EnableCudaGraph);
        AddIfNotNull("tunable_op_enable", TunableOpEnable);
        AddIfNotNull("tunable_op_tuning_enable", TunableOpTuningEnable);
        AddIfNotNull("tunable_op_max_tuning_duration_ms", TunableOpMaxTuningDurationMs);

        return dict;
    }

    public static implicit operator Dictionary<string, string>(CudaConfig config) => config?.ToDictionary() ?? new();
}