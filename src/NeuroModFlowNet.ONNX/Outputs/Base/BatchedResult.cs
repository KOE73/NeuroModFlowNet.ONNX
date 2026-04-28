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

public readonly struct BatchedResultFactory
    : IBatchedResultFactoryPolicy<BatchedResultFactory>
{
    public static BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct
        => new BatchedResult<T>(batchCount, totalCapacity);
}