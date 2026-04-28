namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Box NMS List extractor for FP16.
/// <br/>
/// RU: Экстрактор списка YOLO Box NMS для FP16.
/// </summary>
public class YoloBoxNmsFP16Extractor : YoloBoxNmsFP16ExtractorBase<IDetectionResult<YoloBox_FP16_XYWHSC>>
{
    public override IDetectionResult<YoloBox_FP16_XYWHSC> Extract()
    {
        return GetOutput();
    }
}
