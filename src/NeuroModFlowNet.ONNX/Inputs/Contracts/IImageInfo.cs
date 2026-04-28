namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Interface providing basic image dimensions of the model input.
/// RU: Интерфейс, предоставляющий базовые размеры изображения на входе модели.
/// </summary>
public interface IImageInfo
{
    int Width { get; }
    int Height { get; }
    int Channels { get; }
    int Batch { get; }
}
