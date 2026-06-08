namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP32 to list of 32FC1 Mats.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP32 в список 32FC1 Mat.
/// </summary>
public class PaddleOCRDetFP32_32FC1_SafeListExtractor : PaddleOCRDetFP32_ExtractorBase<List<Mat>>
{
    public override List<Mat> Extract(IOnnxModelOutputs outputs)
    {
        int batchCount = GetBatchCount(outputs);
        var result = new List<Mat>(batchCount);
        for(int i = 0; i < batchCount; i++)
            result.Add(GetOutputAsMat_32FC1_Safe(outputs, i));
        return result;
    }
}
