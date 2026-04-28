namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 (Single Mat).
/// <br/>
/// RU: Экстрактор PaddleOCR Detection для FP16 (одиночный Mat).
/// </summary>
public class PaddleOCRDetFP16_16FC1_SafeExtractor : PaddleOCRDetFP16_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_16FC1_Safe();
}
