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
    public override List<Mat> Extract()
    {
        var result = new List<Mat>(BatchCount);
        for(int i = 0; i < BatchCount; i++)
            result.Add(GetOutputAsMat_16FC1_Safe(i));
        return result;
    }
}
