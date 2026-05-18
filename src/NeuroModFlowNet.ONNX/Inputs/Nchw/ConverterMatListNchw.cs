using NeuroModFlowNet.ONNX.Converters.Algorithms;

namespace NeuroModFlowNet.ONNX.Converters.Nchw;

/// <summary>
/// EN: Generic NCHW converter for batch input (List of Mat). Algorithm is injected via TAlgo struct policy.
/// Designed as a high-performance, low-allocation alternative to <c>CvDnn.BlobFromImages</c> (which is accurate but slower).
/// <para/>
/// RU: Обобщённый NCHW конвертер для батча изображений (List of Mat). Алгоритм внедряется через struct-политику TAlgo.
/// Разработан как высокопроизводительная альтернатива с низким уровнем аллокаций методу <c>CvDnn.BlobFromImages</c> (который точен, но медленнее).
/// </summary>
public class ConverterMatListNchw<TBuf, TAlgo> : ConverterNchwBase<List<Mat>>
    where TBuf : unmanaged
    where TAlgo : struct, IMatListFillAlgorithm<TBuf>
{
    public override string ConverterName => $"MatListNchw<{typeof(TAlgo).Name}>";

    public sealed override void Prepare(List<Mat> images)
    {
        var buffer = Model.GetInputBuffer<TBuf>(Model.PrimaryInputName);
        if(images is null || images.Count == 0) { buffer.Clear(); return; }
        TAlgo.Fill(images, buffer, images.Count, SizeOne, PixelsCount);
    }
}
