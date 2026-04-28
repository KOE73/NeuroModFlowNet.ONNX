namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Структура для представления ориентированного ограничивающего прямоугольника (OBB) с координатами, размерами, углом, классом и оценкой.
/// ВНИМАНИЕ: Последовательность полей и их типы должны строго соответствовать выходу модели, иначе результаты будут некорректными.
/// Проверенно на модели YOLO26 с OBB выходом: [X, Y, W, H, Score, ClassId, Angle] - всего 7 полей.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct YoloObb_FP32_XYWHSCA : IOutAsT<YoloObb>
{
    public readonly float X;
    public readonly float Y;
    public readonly float W;
    public readonly float H;
    public readonly float Score;
    public readonly float Class;
    public readonly float Angle;

    public readonly YoloObb AsStd() => new()
    {
        X = X,
        Y = Y,
        W = W,
        H = H,
        Angle = Angle,
        Score = Score,
        Class = (int)Class
    };

    public override readonly string ToString() => $"[OBB] {X,7:F1}, {Y,7:F1} | {W,6:F1}x{H,6:F1} | Angle: {Angle,5:F2} | Score: {Score,5:P0} | Class: {Class,3}";
}
