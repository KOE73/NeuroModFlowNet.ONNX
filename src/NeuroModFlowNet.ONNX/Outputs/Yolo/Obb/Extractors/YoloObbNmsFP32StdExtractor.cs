namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO OBB NMS Std result extractor for FP32.
/// <br/>
/// RU: Экстрактор стандартного результата YOLO OBB NMS для FP32.
/// </summary>
public class YoloObbNmsFP32StdExtractor : YoloObbNmsFP32ExtractorBase<IDetectionResult<YoloObb>>
{
    public override IDetectionResult<YoloObb> Extract()
    {
        return GetOutputStd();
    }
}
