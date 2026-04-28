using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for PaddleOCR Detection extractors specialized for FP16.
/// <br/>
/// RU: Базовый класс для экстракторов PaddleOCR Detection, специализированный для FP16.
/// </summary>
public abstract class PaddleOCRDetFP16_ExtractorBase<TOut> : PaddleOCRDetExtractorBase<TOut>
{

    protected override void Check()
    {
        var meta = Model.Session.OutputMetadata[Model.PrimaryOutputName];
        if(meta.ElementDataType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP16 extractor requires Float16 (Half).");
    }

    public ReadOnlySpan<Half> GetOutputAsSpan() => Model.GetTensorDataAsSpan<Half>(Model.PrimaryOutputName);

    public unsafe Mat GetOutputAsMat_16FC1_Unsafe(int batchIndex = 0)
    {
        var output = GetOutputAsSpan();
        var imageSpan = output.Slice(batchIndex * OneImageSize, OneImageSize);
        fixed(Half* p = imageSpan)
        {
            return Mat.FromPixelData(OutputImageHeight, OutputImageWidth, MatType.CV_16FC1, (nint)p);
        }
    }

    public Mat GetOutputAsMat_16FC1_Safe(int batchIndex = 0) => GetOutputAsMat_16FC1_Unsafe(batchIndex).Clone();

    public override Mat GetOutputAsMat_8UC1(int batchIndex = 0)
    {
        using var mask32FC1 = GetOutputAsMat_16FC1_Unsafe(batchIndex);
        var mask8UC1 = new Mat();
        mask32FC1.ConvertTo(mask8UC1, MatType.CV_8UC1, 255.0);
        return mask8UC1;
    }
}
