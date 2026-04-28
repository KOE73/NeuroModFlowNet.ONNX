namespace NeuroModFlowNet.ONNX;

/// <summary>
///  Интерфейс 
/// </summary>
public interface IOutAsT<T> where T : struct
{
    T AsStd();
}