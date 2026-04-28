namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Segmentation extractor for FP32 (Single batch).
/// <br/>
/// RU: Экстрактор YOLO Segmentation для FP32 (одиночный батч).
/// </summary>
public class YoloSegFP32SingleExtractor : YoloSegFP32ExtractorBase<YoloSegResult_FP32_Mask32>
{
    protected override void Check()
    {
        base.Check();
        if(BatchCount != 1)
            throw new InvalidOperationException($"Invalid BatchCount for {nameof(YoloSegFP32SingleExtractor)}: BatchCount={BatchCount}. Expected 1.");
    }

    public override YoloSegResult_FP32_Mask32 Extract()
    {
        return GetOutput();
    }
}
