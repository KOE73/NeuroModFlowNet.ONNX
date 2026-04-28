namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// EN: Policy interface for single-image fill algorithms (Mat → buffer).
/// RU: Интерфейс политики заполнения буфера из одного изображения (Mat → buffer).
/// </summary>
public interface IMatFillAlgorithm<TBuf> where TBuf : unmanaged
{
    static abstract unsafe void Fill(Mat image, Span<TBuf> buffer, int pixelsCount);
}
