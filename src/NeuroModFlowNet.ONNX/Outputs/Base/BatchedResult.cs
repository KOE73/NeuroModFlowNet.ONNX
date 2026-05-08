namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN:
/// Stores batched results in a regular managed array allocated with new.
/// This is the simplest implementation, but each instance allocates its own _data array
/// and this memory is later handled by GC.
///
/// RU:
/// Хранит результаты batch в обычном managed-массиве, созданном через new.
/// Это самая простая реализация, но каждый экземпляр выделяет собственный массив _data,
/// и эта память затем обслуживается GC.
/// </summary>
/// <typeparam name="T">Result item type.</typeparam>
public sealed class BatchedResult<T> : BatchedResultBase<T>
    where T : struct
{

    public BatchedResult(int batchCount, int totalCapacity)
        : base(batchCount, totalCapacity)
    {
    }

    protected override T[] AllocateData(int totalCapacity)
        => new T[totalCapacity];
}

/// <summary>
/// EN:
/// Result storage policy that creates BatchedResult{T}.
/// The result object and its _data array are allocated with new and handled by GC.
///
/// RU:
/// Политика хранения результата, создающая BatchedResult{T}.
/// Объект результата и его массив _data выделяются через new и обслуживаются GC.
/// </summary>
public readonly struct BatchedResultFactory
    : IBatchedResultFactoryPolicy<BatchedResultFactory>
{
    public static BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct
        => new BatchedResult<T>(batchCount, totalCapacity);
}
