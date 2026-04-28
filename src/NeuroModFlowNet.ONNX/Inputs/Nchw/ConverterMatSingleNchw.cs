using NeuroModFlowNet.ONNX.Converters.Algorithms;

namespace NeuroModFlowNet.ONNX.Converters.Nchw;

/// <summary>
/// EN: Generic NCHW converter for single Mat input. Algorithm is injected via TAlgo struct policy.
/// RU: Обобщённый NCHW конвертер для одного изображения Mat. Алгоритм внедряется через struct-политику TAlgo.
/// </summary>
public class ConverterMatSingleNchw<TBuf, TAlgo> : ConverterNchwBase<Mat>
    where TBuf : unmanaged
    where TAlgo : struct, IMatFillAlgorithm<TBuf>
{
    public override string ConverterName => $"MatNchw<{typeof(TAlgo).Name}>";

    public sealed override void Prepare(Mat image)
    {
        var buffer = Model.GetInputBuffer<TBuf>(Model.PrimaryInputName);
        if(image.Empty()) { buffer.Clear(); return; }
        TAlgo.Fill(image, buffer, PixelsCount);
    }
}
