using System.Collections.Concurrent;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Global lazy cache for generated ONNX operator sessions.
///
/// RU: Глобальный lazy-кэш для сгенерированных ONNX operator sessions.
/// </summary>
/// <remarks>
/// EN: The cache owns immutable <see cref="OnnxModel"/> instances only. Each caller receives a fresh
/// <see cref="OnnxExecutionContext"/> with its own RunOptions and IoBinding because those objects are mutable native
/// execution state and must not be shared between VM runs.
///
/// RU: Кэш владеет только неизменяемыми экземплярами <see cref="OnnxModel"/>. Каждый вызывающий получает свежий
/// <see cref="OnnxExecutionContext"/> со своими RunOptions и IoBinding, потому что эти объекты являются mutable native
/// состоянием исполнения и не должны разделяться между VM-запусками.
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

        Lazy<OnnxModel> lazyModel = Models.GetOrAdd(
            key,
            static (cacheKey, state) => new Lazy<OnnxModel>(
                () =>
                {
                    byte[] modelBytes = state.BuildModelBytes();
                    return new OnnxModel(modelBytes, cacheKey.Backend, state.Configure, state.DisplayName);
                },
                LazyThreadSafetyMode.ExecutionAndPublication),
            (BuildModelBytes: buildModelBytes, Configure: configure, DisplayName: displayName));

        return new OnnxExecutionContext(lazyModel.Value);
    }

    public static int Count => Models.Count;
}
