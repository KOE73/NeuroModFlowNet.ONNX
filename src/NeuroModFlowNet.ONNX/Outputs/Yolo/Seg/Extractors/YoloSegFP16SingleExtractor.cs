namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Segmentation extractor for Float16 (Single batch).
/// <br/>
/// RU: Экстрактор YOLO Segmentation для Float16 (одиночный батч).
/// </summary>
public class YoloSegFP16SingleExtractor : YoloSegFP16ExtractorBase<YoloSegResult_FP16_Mask32>
{
    protected override void Check()
    {
        base.Check();
        if(BatchCount != 1)
            throw new InvalidOperationException($"Invalid BatchCount for {nameof(YoloSegFP16SingleExtractor)}: BatchCount={BatchCount}. Expected 1.");
    }

    public override YoloSegResult_FP16_Mask32 Extract()
    {
        return GetOutput();
    }
}
