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