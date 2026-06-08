using OpenCvSharp;
using System.Collections.Generic;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 to list of 16FC1 Mats.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP16 в список 16FC1 Mat.
/// </summary>
public class PaddleOCRDetFP16_16FC1_SafeListExtractor : PaddleOCRDetFP16_ExtractorBase<List<Mat>>
{
    public override List<Mat> Extract(IOnnxModelOutputs outputs)
    {
        int batchCount = GetBatchCount(outputs);
        var result = new List<Mat>(batchCount);
        for(int i = 0; i < batchCount; i++)
            result.Add(GetOutputAsMat_16FC1_Safe(outputs, i));
        return result;
    }
}
