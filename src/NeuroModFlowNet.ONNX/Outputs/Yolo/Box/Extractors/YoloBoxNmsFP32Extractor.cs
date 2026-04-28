namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Box NMS List extractor for FP32.
/// <br/>
/// RU: Экстрактор списка YOLO Box NMS для FP32.
/// </summary>
public class YoloBoxNmsFP32Extractor : YoloBoxNmsFP32ExtractorBase<IDetectionResult<YoloBox_FP32_XYWHSC>>
{
    public override IDetectionResult<YoloBox_FP32_XYWHSC> Extract()
    {
        return GetOutput();
    }
}
