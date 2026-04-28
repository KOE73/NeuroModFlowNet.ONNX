namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Interface for input adapters that provide information about the image geometry (Width, Height, etc.).
/// RU: Интерфейс для адаптеров ввода, которые предоставляют информацию о геометрии изображения (Ширина, Высота и т.п.).
/// </summary>
public interface IImageConverter<in TIn> : 
    IInputConverter<TIn>, 
    IImageInfo
    {
    // SetModel inherited from IInputAdapter
}
