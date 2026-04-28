namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Abstract struct for oriented box (OBB). 
/// Структура для хранения данных ориентированного бокса (координаты, размер, угол, класс, оценка) без учета внутренностей нейросети.
/// </summary>
public struct YoloObb : IOutAsT<YoloObb>
{
    public float X;
    public float Y;
    public float W;
    public float H;
    public float Angle;
    public float Score;
    public int Class;

    public readonly YoloObb AsStd() => this;
}
