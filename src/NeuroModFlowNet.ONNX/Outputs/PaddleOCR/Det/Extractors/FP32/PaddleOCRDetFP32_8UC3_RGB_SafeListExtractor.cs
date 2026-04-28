using OpenCvSharp;
using System.Collections.Generic;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: PaddleOCR Detection extractor for FP32 to list of 8UC3 RGB Mats.
/// <br/>
/// RU: Экстрактор PaddleOCR Detection из FP32 в список 8UC3 RGB Mat.
/// </summary>
public class PaddleOCRDetFP32_8UC3_RGB_SafeListExtractor : PaddleOCRDetFP32_ExtractorBase<List<Mat>>
{
    public override List<Mat> Extract()
    {
        var result = new List<Mat>(BatchCount);
        for(int i = 0; i < BatchCount; i++)
            result.Add(GetOutputAsMat_8UC3_RGB(i));
        return result;
    }
}
