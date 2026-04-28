using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for PaddleOCR Detection extractors specialized for FP32.
/// <br/>
/// RU: Базовый класс для экстракторов PaddleOCR Detection, специализированный для FP32.
/// </summary>
public abstract class PaddleOCRDetFP32_ExtractorBase<TOut> : PaddleOCRDetExtractorBase<TOut>
{
    protected override void Check()
    {
        var meta = Model.Session.OutputMetadata[Model.PrimaryOutputName];
        if(meta.ElementDataType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP32 extractor requires Float.");
    }

    public ReadOnlySpan<float> GetOutputAsSpan() => Model.GetTensorDataAsSpan<float>(Model.PrimaryOutputName);

    public unsafe Mat GetOutputAsMat_32FC1_Unsafe(int batchIndex = 0)
    {
        var output = GetOutputAsSpan();
        var imageSpan = output.Slice(batchIndex * OneImageSize, OneImageSize);

        fixed(float* p = imageSpan)
        {
            Mat mat = Mat.FromPixelData(OutputImageHeight, OutputImageWidth, MatType.CV_32FC1, (nint)p);
            return mat;
        }
    }

    public Mat GetOutputAsMat_32FC1_Safe(int batchIndex = 0) => GetOutputAsMat_32FC1_Unsafe(batchIndex).Clone();

    public override Mat GetOutputAsMat_8UC1(int batchIndex = 0)
    {
        using var mask32FC1 = GetOutputAsMat_32FC1_Unsafe(batchIndex);
        var mask8UC1 = new Mat();
        mask32FC1.ConvertTo(mask8UC1, MatType.CV_8UC1, 255.0);
        return mask8UC1;
    }

}
