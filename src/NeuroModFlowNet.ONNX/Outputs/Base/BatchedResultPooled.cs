using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Вспомогательный класс.
/// Организует хранение результатов детекции в виде единого плотного массива с индексами для каждого батча.
/// Основное преимущество, экономи GC. 
/// При большом количестве детекций (например, при сегментации) позволяет избежать большого количества мелких массивов для каждого батча.
/// 
/// TODO: посмотреть как пулы использоватью
/// </summary>
/// <typeparam name="T"></typeparam>
using System.Buffers;
using System.Runtime.CompilerServices;

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


public readonly struct BatchedResultPooledFactory
    : IBatchedResultFactoryPolicy<BatchedResultPooledFactory>
{
    public static BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct
        => new BatchedResultPooled<T>(batchCount, totalCapacity);
}