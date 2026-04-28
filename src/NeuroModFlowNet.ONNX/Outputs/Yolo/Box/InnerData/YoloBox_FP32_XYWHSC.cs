namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Packed structure for a single bounding box detection result with NMS (from model output).
/// <br/>
/// RU: Структура для одного результата детекции прямоугольного бокса с NMS (из выхода модели).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct YoloBox_FP32_XYWHSC : IOutAsT<YoloBox>
{
    public float X;
    public float Y;
    public float W;
    public float H;
    public float Score;
    public float Class;

    //readonly float IYoloBoxSource.X => X;
    //readonly float IYoloBoxSource.Y => Y;
    //readonly float IYoloBoxSource.W => W;
    //readonly float IYoloBoxSource.H => H;
    //readonly float IYoloBoxSource.Score => Score;
    //readonly float IYoloBoxSource.Class => Class;


    //public readonly YoloBox ToUniversal() => new() { X = X, Y = Y, W = W, H = H, Score = Score, Class = (int)Class };
    
    public readonly YoloBox AsStd() => new YoloBox() { X = X, Y = Y, W = W, H = H, Score = Score, Class = (int)Class };

    public override readonly string ToString() => $"[Box] {X,7:F1}, {Y,7:F1} | {W,6:F1}x{H,6:F1} | Score: {Score,5:P0} | Class: {Class,3}";
}
