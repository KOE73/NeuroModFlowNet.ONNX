namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP32 to 32FC1 Mat with cloning.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP32 в 32FC1 Mat с клонированием.
/// </summary>
public class PaddleOCRDetFP32_32FC1_SafeExtractor : PaddleOCRDetFP32_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_32FC1_Safe();
}
