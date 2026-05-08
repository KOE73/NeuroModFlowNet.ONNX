namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN:
/// Stores results for all batch items in one dense array.
/// Each batch stores only its range in the shared array: Offset + Length.
/// This avoids many small per-batch arrays and reduces GC pressure.
///
/// RU:
/// Хранит результаты всех элементов batch в одном плотном массиве.
/// Для каждого batch сохраняется только диапазон в общем массиве: Offset + Length.
/// Это позволяет избежать множества мелких массивов на каждый batch и снижает нагрузку на GC.
/// </summary>
/// <remarks>
/// <code>
///   _ranges:
///   Batch 0          Batch 1      Batch 2
///   O=0 L=3          O=3 L=2      O=5 L=4
///   | I0 I1 I2 |     | I0 I1 |    | I0 I1 I2 I3 |
///
///   _data:
///   | B0.I0 B0.I1 B0.I2 | B1.I0 B1.I1 | B2.I0 B2.I1 B2.I2 B2.I3 |
/// </code>
/// </remarks>
/// <typeparam name="T">The value type of the items contained in the batched result.</typeparam>
public abstract class BatchedResultBase<T> : IBatchedResult, IDetectionResult<T>
    where T : struct
{
    protected T[] _data;
    readonly (int Offset, int Length)[] _ranges;

    int _currentBatch = -1;
    int _currentWriteIndex = 0;

    public int BatchCount { get; }

    protected BatchedResultBase(int batchCount, int totalCapacity)
    {
        BatchCount = batchCount;
        _ranges = new (int, int)[batchCount];
        _data = AllocateData(totalCapacity);
    }

    protected abstract T[] AllocateData(int totalCapacity);

    protected virtual void ReleaseData(T[] data)
    {
    }

    public void Add(T item)
    {
        _data[_currentWriteIndex++] = item;
        _ranges[_currentBatch].Length++;
    }

    public bool MoveNext()
    {
        if(_currentBatch + 1 >= BatchCount)
            return false;

        _currentBatch++;
        _ranges[_currentBatch] = (_currentWriteIndex, 0);
        return true;
    }

    public void SetBatch(int batchIndex, ReadOnlySpan<T> data)
    {
        _ranges[batchIndex] = (_currentWriteIndex, data.Length);
        data.CopyTo(_data.AsSpan(_currentWriteIndex));
        _currentWriteIndex += data.Length;
    }

    public ReadOnlySpan<T> GetBatch(int batchIndex)
    {
        var (offset, length) = _ranges[batchIndex];
        return _data.AsSpan(offset, length);
    }

    public int GetResultCount(int batchIndex = 0)
    {
        return _ranges[batchIndex].Length;
    }

    public T GetResult(int index, int batchIndex = 0)
    {
        return _data[_ranges[batchIndex].Offset + index];
    }

    protected void DisposeCore()
    {
        var data = _data;
        _data = Array.Empty<T>();

        if(data.Length != 0)
            ReleaseData(data);
    }
}
