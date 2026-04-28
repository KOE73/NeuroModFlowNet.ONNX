namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP32 to 8UC3 RGB Mat.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP32 в 8UC3 RGB Mat.
/// </summary>
public class PaddleOCRDetFP32_8UC3_RGBExtractor : PaddleOCRDetFP32_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_8UC3_RGB();
}
