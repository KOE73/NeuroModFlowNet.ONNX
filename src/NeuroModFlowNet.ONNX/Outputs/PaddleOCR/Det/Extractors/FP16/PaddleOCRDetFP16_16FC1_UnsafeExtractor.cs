namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 to 16FC1 Mat without cloning.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP16 в 16FC1 Mat без клонирования.
/// </summary>
public class PaddleOCRDetFP16_16FC1_UnsafeExtractor : PaddleOCRDetFP16_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_16FC1_Unsafe();
}
