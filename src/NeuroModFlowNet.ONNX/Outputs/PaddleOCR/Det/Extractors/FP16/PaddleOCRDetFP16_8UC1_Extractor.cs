namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 to 8UC1 Mat.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP16 в 8UC1 Mat.
/// </summary>
public class PaddleOCRDetFP16_8UC1_Extractor : PaddleOCRDetFP16_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_8UC1();
}
