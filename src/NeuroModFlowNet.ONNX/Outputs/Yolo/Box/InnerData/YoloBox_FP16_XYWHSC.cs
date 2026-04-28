
namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Packed structure for a single bounding box detection result with NMS (Half version from model output).
/// <br/>
/// RU: Структура для одного результата детекции прямоугольного бокса с NMS (FP16 версия из выхода модели).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct YoloBox_FP16_XYWHSC : IOutAsT<YoloBox>
{
    public Half X;
    public Half Y;
    public Half W;
    public Half H;
    public Half Score;
    public Half Class;

    //readonly float IYoloBoxSource.X => (float)X;
    //readonly float IYoloBoxSource.Y => (float)Y;
    //readonly float IYoloBoxSource.W => (float)W;
    //readonly float IYoloBoxSource.H => (float)H;
    //readonly float IYoloBoxSource.Score => (float)Score;
    //readonly float IYoloBoxSource.Class => (float)Class;

    
    // Явная реализация, чтобы не засорять саму структуру
    //public readonly YoloBox ToUniversal() => new()
    public readonly YoloBox AsStd() => new YoloBox()
    {
        X = (float)X,
        Y = (float)Y,
        W = (float)W,
        H = (float)H,
        Score = (float)Score,
        Class = (int)Class
    };


    public override readonly string ToString() => $"[BoxFP16] {(float)X,7:F1}, {(float)Y,7:F1} | {(float)W,6:F1}x{(float)H,6:F1} | Score: {(float)Score,5:P0} | Class: {(float)Class,3}";
}
