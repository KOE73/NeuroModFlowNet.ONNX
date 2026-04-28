using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX;

// ------------------------------------------------------------
// УНИВЕРСАЛЬНЫЙ ВАРИАНТ ДЛЯ ЛЮБОГО ЧИСЛА ТОЧЕК
// Это не blittable struct фиксированного размера, а view поверх ReadOnlySpan<float>.
// Подходит для любой pose-модели:
// [X, Y, W, H, Score, ClassId, K0.X, K0.Y, K0.V, ..., KN.V]
// ------------------------------------------------------------

public readonly ref struct YoloPose_FP32_RowView
{
    private readonly ReadOnlySpan<float> _row;

    public YoloPose_FP32_RowView(ReadOnlySpan<float> row, int keypointsCount)
    {
        Debug.Assert(keypointsCount >= 0);

        int expectedLength = 6 + keypointsCount * 3;
        if(row.Length < expectedLength)
            throw new ArgumentException("Pose row is shorter than expected.", nameof(row));

        _row = row;
        KeypointsCount = keypointsCount;
    }

    public int KeypointsCount { get; }

    // Полная длина одной записи в float.
    public int Stride => 6 + KeypointsCount * 3;

    public float X => _row[0];
    public float Y => _row[1];
    public float W => _row[2];
    public float H => _row[3];
    public float Score => _row[4];
    public float ClassId => _row[5];

    // Сырой интерливнутый span вида:
    // [K0.X, K0.Y, K0.V, K1.X, K1.Y, K1.V, ...]
    public ReadOnlySpan<float> KeypointsInterleavedXYV => _row.Slice(6, KeypointsCount * 3);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public YoloPose_FP32_Keypoint_XYV GetKeypoint(int index)
    {
        Debug.Assert((uint)index < (uint)KeypointsCount);

        int offset = 6 + index * 3;
        return new YoloPose_FP32_Keypoint_XYV(
            _row[offset],
            _row[offset + 1],
            _row[offset + 2]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public YoloPose_FP32_Keypoint_XYV[] ToKeypointsArray()
    {
        // Здесь данные уже лежат подряд как:
        // [K0.X, K0.Y, K0.V, K1.X, K1.Y, K1.V, ...]
        // Поэтому можно безопасно переинтерпретировать float-span как span keypoint-struct-ов
        // и один раз скопировать в новый managed-массив.
        return MemoryMarshal
            .Cast<float, YoloPose_FP32_Keypoint_XYV>(KeypointsInterleavedXYV)
            .ToArray();
    }

    public override string ToString()
        => $"Box=({X:F1},{Y:F1},{W:F1},{H:F1}) Score={Score:F3} Class={ClassId:F0} Kpts={KeypointsCount}";
}

