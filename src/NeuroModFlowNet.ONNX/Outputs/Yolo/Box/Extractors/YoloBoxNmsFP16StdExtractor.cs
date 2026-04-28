namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Box NMS Single result extractor for FP16.
/// <br/>
/// RU: Экстрактор одного результата YOLO Box NMS для FP16.
/// </summary>
public class YoloBoxNmsFP16StdExtractor : YoloBoxNmsFP16ExtractorBase<IDetectionResult<YoloBox>>
{
    public override IDetectionResult<YoloBox> Extract()
    {
        return GetOutputStd();
    }
}
