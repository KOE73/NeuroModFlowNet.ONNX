using OpenCvSharp;
using System.Collections.Generic;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP16 to list of 8UC3 RGB Mats.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP16 в список 8UC3 RGB Mat.
/// </summary>
public class PaddleOCRDetFP16_8UC3_RGB_SafeListExtractor : PaddleOCRDetFP16_ExtractorBase<List<Mat>>
{
    public override List<Mat> Extract(IOnnxModelOutputs outputs)
    {
        int batchCount = GetBatchCount(outputs);
        var result = new List<Mat>(batchCount);
        for(int i = 0; i < batchCount; i++)
            result.Add(GetOutputAsMat_8UC3_RGB(outputs, i));
        return result;
    }
}
