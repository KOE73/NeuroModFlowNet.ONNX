using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Полная конфигурация TensorRT Execution Provider.
/// Свойства сгруппированы по назначению. Если свойство null — параметр не передается (default ORT).
/// </summary>
public sealed class TrtConfig : ExecutionProviderConfig
{
    public override InferenceBackend InferenceBackend => InferenceBackend.TensorRt;

    // --- ОСНОВНЫЕ НАСТРОЙКИ ---
    public double? MaxWorkspaceSizeGb { get; set; }
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed per partition
    ///  for TensorRT engine building.
    /// </summary>
    public int? MaxPartitionIterations { get; set; }
    public int? MinSubgraphSize { get; set; }

    // --- ТОЧНОСТЬ И СКОРОСТЬ (PRECISION) ---
    public bool? EnableFp16 { get; set; }
    public bool? EnableBf16 { get; set; }
    public bool? EnableInt8 { get; set; }
    public string? Int8CalibrationTableName { get; set; }
    public bool? Int8UseNativeCalibrationTable { get; set; }
    public bool? LayerNormFp32Fallback { get; set; }
    /// <summary>
    /// Gets or sets a value indicating whether sparsity optimizations are enabled.
    /// </summary>
    /// <remarks>When enabled, operations may leverage sparse data structures or algorithms to improve
    /// performance and reduce memory usage for data with many zero or default values. If set to <see
    /// langword="null"/>, the default behavior is applied.</remarks>
    public bool? EnableSparsity { get; set; }

    // --- КЭШИРОВАНИЕ (ENGINE CACHING) ---
    public bool? EnableEngineCache { get; set; }
    public string? EngineCachePath { get; set; }
    public string? EngineCachePrefix { get; set; }

    // --- ОПТИМИЗАЦИЯ СБОРКИ (BUILDER) ---
    public int? BuilderOptimizationLevel { get; set; }
    public bool? ForceSequentialEngineBuild { get; set; }
    public string? TacticSources { get; set; }  // "CUBLAS,CUBLAS_LT,CUDNN,EDGE_MASK_CONVOLUTIONS"

    // --- ПРОДВИНУТЫЕ ТЕХНОЛОГИИ (DLA & LOGGING) ---
    public bool? EnableDla { get; set; }
    public int? DlaCore { get; set; }
    public bool? DetailedBuildLog { get; set; }
    public bool? DumpSubgraphs { get; set; }
    public bool? ExternalAllocatorEnable { get; set; }
    public bool? TimingCacheEnable { get; set; }

    // --- ДИНАМИЧЕСКИЕ ФОРМЫ(DYNAMIC SHAPES) ---
    // Формат: "input_name:dim0xdim1x...,input_name2:dim0xdim1"
    public string? ProfileMinShapes { get; set; }
    public string? ProfileMaxShapes { get; set; }
    public string? ProfileOptShapes { get; set; }

    public static TrtConfig FromDefaultOptions(OrtTensorRTProviderOptions trtOptions) =>
        FromDefaultOptions(trtOptions.GetOptions());

    /// <summary>
    /// Создает конфиг из сырой строки параметров ONNX Runtime.
    /// </summary>
    /// <param name="rawOptions">Строка вида "device_id=0;trt_fp16_enable=1;..."</param>
    public static TrtConfig FromDefaultOptions(string rawOptions)
    {
        var config = new TrtConfig();
        if(string.IsNullOrWhiteSpace(rawOptions)) return config;

        // Разрезаем строку на пары ключ-значение
        var dict = rawOptions.Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(part => part.Split('=', 2))
            .Where(part => part.Length == 2)
            .ToDictionary(sp => sp[0].Trim(), sp => sp[1].Trim());

        // Вспомогательные функции парсинга
        bool? ParseBool(string key) => dict.TryGetValue(key, out var v) ? (v == "1") : null;
        int? ParseInt(string key) => dict.TryGetValue(key, out var v) && int.TryParse(v, out var res) ? res : null;
        string? GetStr(string key) => dict.TryGetValue(key, out var v) ? v : null;

        // --- ОСНОВНЫЕ НАСТРОЙКИ ---
        config.DeviceId = ParseInt("device_id") ?? 0;
        if(dict.TryGetValue("trt_max_workspace_size", out var wsStr) && long.TryParse(wsStr, out var wsBytes))
        {
            config.MaxWorkspaceSizeGb = wsBytes / (1024.0 * 1024.0 * 1024.0);
        }
        config.MaxPartitionIterations = ParseInt("trt_max_partition_iterations");
        config.MinSubgraphSize = ParseInt("trt_min_subgraph_size");

        // --- ТОЧНОСТЬ И СКОРОСТЬ (PRECISION) ---
        config.EnableFp16 = ParseBool("trt_fp16_enable");
        config.EnableBf16 = ParseBool("trt_bf16_enable");
        config.EnableInt8 = ParseBool("trt_int8_enable");
        config.Int8CalibrationTableName = GetStr("trt_int8_calibration_table_name");
        config.Int8UseNativeCalibrationTable = ParseBool("trt_int8_use_native_calibration_table");
        config.LayerNormFp32Fallback = ParseBool("trt_layer_norm_fp32_fallback");
        config.EnableSparsity = ParseBool("trt_sparsity_enable");

        // --- КЭШИРОВАНИЕ (ENGINE CACHING) ---
        config.EnableEngineCache = ParseBool("trt_engine_cache_enable");
        config.EngineCachePath = GetStr("trt_engine_cache_path") ?? config.EngineCachePath; // сохраняем наш дефолт, если там пусто
        config.EngineCachePrefix = GetStr("trt_engine_cache_prefix");

        // --- ОПТИМИЗАЦИЯ СБОРКИ (BUILDER) ---
        config.BuilderOptimizationLevel = ParseInt("trt_builder_optimization_level");
        config.ForceSequentialEngineBuild = ParseBool("trt_force_sequential_engine_build");
        config.TacticSources = GetStr("trt_tactic_sources");

        // --- ПРОДВИНУТЫЕ ТЕХНОЛОГИИ (DLA & LOGGING) ---
        config.EnableDla = ParseBool("trt_dla_enable");
        config.DlaCore = ParseInt("trt_dla_core");
        config.DetailedBuildLog = ParseBool("trt_detailed_build_log");
        config.DumpSubgraphs = ParseBool("trt_dump_subgraphs") ?? false;
        config.ExternalAllocatorEnable = ParseBool("trt_external_allocator_enable");
        config.TimingCacheEnable = ParseBool("trt_timing_cache_enable");

        config.ProfileMinShapes = GetStr("trt_profile_min_shapes");
        config.ProfileMaxShapes = GetStr("trt_profile_max_shapes");
        config.ProfileOptShapes = GetStr("trt_profile_opt_shapes");

        // Добавь остальные поля по аналогии...

        return config;
    }

    /// <summary>
    /// Формирует словарь только из тех параметров, которые были установлены (не null).
    /// </summary>
    public override Dictionary<string, string> ToDictionary()
    {
        var dict = new Dictionary<string, string>();

        // Вспомогательный метод для компактности
        void AddIfNotNull(string key, object? value)
        {
            if(value is null) return;

            string stringValue = value is bool b ? (b ? "1" : "0") : value.ToString()!;

            if(string.IsNullOrWhiteSpace(stringValue)) return;

            dict[key] = stringValue;

            //if(value is bool b)
            //{
            //    dict[key] = b ? "1" : "0";
            //    return;
            //}

            //if(value.ToString() is string s )
            //{
            //    dict[key] = s;
            //}
        }

        // Проверяем и создаем директорию кэша заранее
        if(EnableEngineCache == true && !string.IsNullOrEmpty(EngineCachePath))
        {
            try { Directory.CreateDirectory(EngineCachePath); } catch { /* Логировать ошибку доступа */ }
        }

        // Заполнение словаря
        AddIfNotNull("device_id", DeviceId);

        if(MaxWorkspaceSizeGb.HasValue)
            dict["trt_max_workspace_size"] = ((long)(MaxWorkspaceSizeGb.Value * 1024 * 1024 * 1024)).ToString();

        AddIfNotNull("trt_max_partition_iterations", MaxPartitionIterations);
        AddIfNotNull("trt_min_subgraph_size", MinSubgraphSize);

        // Precision
        AddIfNotNull("trt_fp16_enable", EnableFp16);
        AddIfNotNull("trt_bf16_enable", EnableBf16);
        AddIfNotNull("trt_int8_enable", EnableInt8);
        AddIfNotNull("trt_int8_calibration_table_name", Int8CalibrationTableName);
        AddIfNotNull("trt_int8_use_native_calibration_table", Int8UseNativeCalibrationTable);
        AddIfNotNull("trt_sparsity_enable", EnableSparsity);
        AddIfNotNull("trt_layer_norm_fp32_fallback", LayerNormFp32Fallback);

        // Caching
        AddIfNotNull("trt_engine_cache_enable", EnableEngineCache);
        AddIfNotNull("trt_engine_cache_path", EngineCachePath);
        AddIfNotNull("trt_engine_cache_prefix", EngineCachePrefix);

        // Builder
        AddIfNotNull("trt_builder_optimization_level", BuilderOptimizationLevel);
        AddIfNotNull("trt_force_sequential_engine_build", ForceSequentialEngineBuild);
        AddIfNotNull("trt_tactic_sources", TacticSources);

        // Advanced
        AddIfNotNull("trt_dla_enable", EnableDla);
        AddIfNotNull("trt_dla_core", DlaCore);
        AddIfNotNull("trt_detailed_build_log", DetailedBuildLog);
        AddIfNotNull("trt_dump_subgraphs", DumpSubgraphs);
        AddIfNotNull("trt_external_allocator_enable", ExternalAllocatorEnable);
        AddIfNotNull("trt_timing_cache_enable", TimingCacheEnable);

        // Dynamic Shapes Profiles
        AddIfNotNull("trt_profile_min_shapes", ProfileMinShapes);
        AddIfNotNull("trt_profile_max_shapes", ProfileMaxShapes);
        AddIfNotNull("trt_profile_opt_shapes", ProfileOptShapes);

        //Min: pixel_sequence: 1x1x512x512x3(минимум 1 картинка в батче).
        //Max: pixel_sequence: 16x1x512x512x3(максимум 16 картинок).
        //Opt: pixel_sequence: 8x1x512x512x3(наиболее частый размер для оптимизации ядер Blackwell).

        return dict;
    }

    public static implicit operator Dictionary<string, string>(TrtConfig config)
    {
        return config?.ToDictionary() ?? new Dictionary<string, string>();
    }
}