using System.Runtime.CompilerServices;


namespace NeuroModFlowNet.ONNX;    


// ------------------------------------------------------------
// УНИВЕРСАЛЬНЫЙ ENUMERABLE/VIEW ПО ВСЕМ DETECTION
// Без аллокаций, без копирования, просто режет исходный float span на строки.
// ------------------------------------------------------------

public readonly ref struct YoloPose_FP32_RowsView
{
    private readonly ReadOnlySpan<float> _data;

    public YoloPose_FP32_RowsView(ReadOnlySpan<float> data, int keypointsCount)
    {
        Debug.Assert(keypointsCount >= 0);

        int stride = 6 + keypointsCount * 3;
        if(stride <= 0 || data.Length % stride != 0)
            throw new ArgumentException("Pose tensor length is not divisible by row stride.", nameof(data));

        _data = data;
        KeypointsCount = keypointsCount;
        Stride = stride;
        Count = data.Length / stride;
    }

    public int KeypointsCount { get; }
    public int Stride { get; }
    public int Count { get; }

    public YoloPose_FP32_RowView this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert((uint)index < (uint)Count);
            return new YoloPose_FP32_RowView(_data.Slice(index * Stride, Stride), KeypointsCount);
        }
    }
}

