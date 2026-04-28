namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Abstract struct for box. 
/// Структура для хранения данных бокса (координаты, размер, класс, оценка) без учета внутренностей нейросети.
/// </summary>
public struct YoloBox : IOutAsT<YoloBox>
{
    public float X;
    public float Y;
    public float W;
    public float H;
    public float Score;
    public int Class;

    public readonly YoloBox AsStd() => this;
}
