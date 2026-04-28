namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP32 to 8UC1 Mat.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP32 в 8UC1 Mat.
/// </summary>
public class PaddleOCRDetFP32_8UC1_Extractor : PaddleOCRDetFP32_ExtractorBase<Mat>
{
    public override Mat Extract() => GetOutputAsMat_8UC1();
}
