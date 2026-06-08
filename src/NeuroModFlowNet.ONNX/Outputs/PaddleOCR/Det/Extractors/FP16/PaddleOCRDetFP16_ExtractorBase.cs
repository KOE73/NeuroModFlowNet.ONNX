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
        var elementType = Model.GetOutputElementType(Model.PrimaryOutputName);
        if(elementType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model produces {elementType}, but FP16 extractor requires Float16 (Half).");
    }

    public ReadOnlySpan<Half> GetOutputAsSpan(IOnnxModelOutputs outputs) => outputs.GetTensorDataAsSpan<Half>(Model.PrimaryOutputName);

    public unsafe Mat GetOutputAsMat_16FC1_Unsafe(IOnnxModelOutputs outputs, int batchIndex = 0)
    {
        var output = GetOutputAsSpan(outputs);
        int height = GetOutputImageHeight(outputs);
        int width = GetOutputImageWidth(outputs);
        int imageSize = height * width;
        var imageSpan = output.Slice(batchIndex * imageSize, imageSize);
        fixed(Half* p = imageSpan)
        {
            return Mat.FromPixelData(height, width, MatType.CV_16FC1, (nint)p);
        }
    }

    public Mat GetOutputAsMat_16FC1_Safe(IOnnxModelOutputs outputs, int batchIndex = 0) => GetOutputAsMat_16FC1_Unsafe(outputs, batchIndex).Clone();

    public override Mat GetOutputAsMat_8UC1(IOnnxModelOutputs outputs, int batchIndex = 0)
    {
        using var mask32FC1 = GetOutputAsMat_16FC1_Unsafe(outputs, batchIndex);
        var mask8UC1 = new Mat();
        mask32FC1.ConvertTo(mask8UC1, MatType.CV_8UC1, 255.0);
        return mask8UC1;
    }
}
