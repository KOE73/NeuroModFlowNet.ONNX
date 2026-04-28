namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 to 8UC3 RGB Mat.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP16 в 8UC3 RGB Mat.
/// </summary>
public class PaddleOCRDetFP16_8UC3_RGBExtractor : PaddleOCRDetFP16_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_8UC3_RGB();
}
