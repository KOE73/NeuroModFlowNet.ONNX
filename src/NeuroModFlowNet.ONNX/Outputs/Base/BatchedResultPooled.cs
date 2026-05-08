namespace NeuroModFlowNet.ONNX;

using System.Buffers;
using System.Runtime.CompilerServices;

/// <summary>
/// EN:
/// Stores batched results in an array rented from ArrayPool instead of allocating _data with new.
/// This reduces managed allocations and GC pressure for large or frequently created result buffers.
/// The rented array must be returned to the pool through Dispose.
///
/// RU:
/// Хранит результаты batch в массиве, взятом из ArrayPool, а не созданном через new для _data.
/// Это снижает количество managed-аллокаций и нагрузку на GC при больших или часто создаваемых буферах результатов.
/// Арендованный массив должен быть возвращен в пул через Dispose.
/// </summary>
/// <remarks>
/// EN:
/// The buffer is cleared before return only when T contains references.
///
/// RU:
/// Буфер очищается перед возвратом только если T содержит ссылки.
/// </remarks>
/// <typeparam name="T">Result item type.</typeparam>
public sealed class BatchedResultPooled<T> : BatchedResultBase<T>, IDisposable
    where T : struct
{
    static readonly bool ClearOnReturn = RuntimeHelpers.IsReferenceOrContainsReferences<T>();
    bool _disposed;

    public BatchedResultPooled(int batchCount, int totalCapacity)
        : base(batchCount, totalCapacity)
    {
    }

    protected override T[] AllocateData(int totalCapacity) => ArrayPool<T>.Shared.Rent(totalCapacity);

    protected override void ReleaseData(T[] data) => ArrayPool<T>.Shared.Return(data, clearArray: ClearOnReturn);

    public void Dispose()
    {
        if(_disposed)
            return;

        _disposed = true;
        DisposeCore();
    }
}


/// <summary>
/// EN:
/// Result storage policy that creates BatchedResultPooled{T}.
/// The result object is allocated with new, but its _data array is rented from ArrayPool.
///
/// RU:
/// Политика хранения результата, создающая BatchedResultPooled{T}.
/// Объект результата выделяется через new, но его массив _data берется из ArrayPool.
/// </summary>
public readonly struct BatchedResultPooledFactory
    : IBatchedResultFactoryPolicy<BatchedResultPooledFactory>
{
    public static BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct
        => new BatchedResultPooled<T>(batchCount, totalCapacity);
}
