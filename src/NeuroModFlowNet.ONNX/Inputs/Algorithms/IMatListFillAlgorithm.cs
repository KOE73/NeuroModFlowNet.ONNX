namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// EN: Policy interface for batch fill algorithms (List of Mat → buffer).
/// RU: Интерфейс политики заполнения буфера из списка изображений (List of Mat → buffer).
/// </summary>
public interface IMatListFillAlgorithm<TBuf> where TBuf : unmanaged
{
    static abstract unsafe void Fill(List<Mat> mats, Span<TBuf> buffer,
        int matsCount, int sizeOne, int pixelsCount);
}
