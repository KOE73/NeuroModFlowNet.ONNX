using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Global, lazy cache of runtime-generated ONNX operator kernels (Crop, Resize, Undistort, ROI batches, concat,
/// identity upload) keyed by operation kind, exact shape/semantics, backend and provider options. Only the immutable
/// <see cref="OnnxModel"/> is shared; every caller gets its own <see cref="OnnxExecutionContext"/> (mutable IoBinding
/// and RunOptions).
///
/// RU: Глобальный lazy-кэш runtime-generated ONNX kernels (Crop, Resize, Undistort, ROI-батчи, concat, identity
/// upload) по ключу: тип операции, точная форма/семантика, backend и provider options. Общим является только
/// immutable <see cref="OnnxModel"/>; каждый вызывающий получает свой <see cref="OnnxExecutionContext"/>
/// (mutable IoBinding и RunOptions).
/// </summary>
/// <remarks>
/// EN: TensorRT engine cache policy lives here, not in the operators. A generated graph is tiny and its TensorRT
/// engine build is what costs seconds at warmup, so engines are persisted. The on-disk identity must equal the kernel
/// identity: the engine cache prefix is a hash of the full <see cref="RuntimeOnnxOperatorKernelKey"/>, so two graphs
/// that differ only in shape, padding or matrix values can never pick up each other's engine (all generated graphs
/// share the same ONNX graph name, which is why the provider's own hashing is not enough).
///
/// RU: Политика engine cache TensorRT задаётся здесь, а не в операторах. Сгенерированный граф крошечный, а секунды на
/// warmup стоит именно сборка engine, поэтому engines сохраняются на диск. Идентичность на диске обязана совпадать с
/// идентичностью kernel-а: префикс engine cache это хэш полного <see cref="RuntimeOnnxOperatorKernelKey"/>, поэтому
/// два графа, отличающиеся только формой, padding или значениями матриц, никогда не подхватят чужой engine (все
/// сгенерированные графы носят одно имя ONNX-графа, поэтому хэширования самого provider-а недостаточно).
/// </remarks>
public static class RuntimeOnnxOperatorKernelCache
{
    static readonly ConcurrentDictionary<RuntimeOnnxOperatorKernelKey, Lazy<OnnxModel>> Models = new();

    public static OnnxExecutionContext CreateContext(
        RuntimeOnnxOperatorKernelKey key,
        Func<byte[]> buildModelBytes,
        Action<ExecutionProviderConfig>? configure,
        string displayName)
    {
        ArgumentNullException.ThrowIfNull(buildModelBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);

        if(SessionSharing == RuntimeKernelSessionSharing.PerCaller)
        {
            // Own session per caller; the engine identity (cache prefix) is still the kernel key, so the TensorRT engine
            // file built once is reused by every session. The context owns the model and releases it on dispose.
            byte[] modelBytes = buildModelBytes();
            var model = new OnnxModel(modelBytes, key.Backend, config => ConfigureProvider(config, key, configure), displayName);
            return new OnnxExecutionContext(model, ownsModel: true);
        }

        Lazy<OnnxModel> lazyModel = Models.GetOrAdd(
            key,
            static (cacheKey, state) => new Lazy<OnnxModel>(
                () =>
                {
                    byte[] modelBytes = state.BuildModelBytes();
                    return new OnnxModel(
                        modelBytes,
                        cacheKey.Backend,
                        config => ConfigureProvider(config, cacheKey, state.Configure),
                        state.DisplayName);
                },
                LazyThreadSafetyMode.ExecutionAndPublication),
            (BuildModelBytes: buildModelBytes, Configure: configure, DisplayName: displayName));

        return new OnnxExecutionContext(lazyModel.Value);
    }

    /// <summary>
    /// EN: Process-wide session policy, set once at startup before any operator initialises. <see cref="RuntimeKernelSessionSharing.Shared"/>
    /// (default) shares one session per kernel key between all callers. <see cref="RuntimeKernelSessionSharing.PerCaller"/>
    /// gives every operator instance its own session: needed when many controllers run the same kernel concurrently,
    /// because runs of one ORT/TensorRT session are serialized (measured: 10 NVDEC streams 175 fps shared vs 1280 fps
    /// per caller on RTX 5090).
    ///
    /// RU: Политика сессий на процесс, задаётся один раз при старте до инициализации операторов.
    /// <see cref="RuntimeKernelSessionSharing.Shared"/> (по умолчанию) делит одну сессию на ключ kernel-а между всеми.
    /// <see cref="RuntimeKernelSessionSharing.PerCaller"/> даёт каждому экземпляру операции свою сессию: нужно, когда
    /// много контроллеров одновременно выполняют один kernel, потому что запуски одной сессии ORT/TensorRT идут по
    /// очереди (замер: 10 потоков NVDEC 175 fps при общей сессии против 1280 fps при своих, RTX 5090).
    /// </summary>
    public static RuntimeKernelSessionSharing SessionSharing { get; set; } = RuntimeKernelSessionSharing.Shared;

    public static int Count => Models.Count;

    /// <summary>
    /// Stable engine cache prefix for a kernel key: the same kernel on the same machine reuses its engine across
    /// processes, a kernel with any other shape or parameter gets a different file.
    /// </summary>
    public static string GetEngineCachePrefix(in RuntimeOnnxOperatorKernelKey key)
    {
        string identity = string.Join('|', key.OperationKind, key.SemanticKey, key.Backend, key.ProviderOptionsKey);
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(Encoding.UTF8.GetBytes(identity), hash);
        return $"nmfn_op_{SanitizeKind(key.OperationKind)}_{Convert.ToHexString(hash[..12]).ToLowerInvariant()}";
    }

    static void ConfigureProvider(ExecutionProviderConfig config, RuntimeOnnxOperatorKernelKey key, Action<ExecutionProviderConfig>? configure)
    {
        configure?.Invoke(config);

        if(config is not TrtConfig trtConfig)
            return;

        trtConfig.EnableEngineCache = true;
        trtConfig.EngineCachePath = TrtConfigDefaults.GetEngineCachePath();
        trtConfig.EngineCachePrefix = GetEngineCachePrefix(key);
        trtConfig.TimingCacheEnable = true;
    }

    static string SanitizeKind(string operationKind)
    {
        var builder = new StringBuilder(operationKind.Length);
        foreach(char character in operationKind)
            builder.Append(char.IsLetterOrDigit(character) ? char.ToLowerInvariant(character) : '_');

        return builder.ToString();
    }
}
