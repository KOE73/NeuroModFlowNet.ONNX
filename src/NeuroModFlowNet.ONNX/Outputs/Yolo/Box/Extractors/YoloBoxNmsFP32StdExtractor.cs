namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Box NMS Single result extractor for FP32.
/// <br/>
/// RU: Экстрактор одного результата YOLO Box NMS для FP32.
/// </summary>
public class YoloBoxNmsFP32StdExtractor : YoloBoxNmsFP32ExtractorBase<IDetectionResult<YoloBox>>
{
    public override IDetectionResult<YoloBox> Extract()
    {
        return GetOutputStd();
    }
}
