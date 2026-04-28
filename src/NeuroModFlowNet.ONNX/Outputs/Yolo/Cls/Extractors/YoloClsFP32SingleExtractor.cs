namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Classification Single result extractor for FP32.
/// <br/>
/// RU: Экстрактор одного результата классификации YOLO для FP32.
/// </summary>
public class YoloClsFP32SingleExtractor : YoloClsFP32ExtractorBase<YoloCls>
{
    protected override void Check()
    {
        base.Check();
        if(BatchCount != 1)
            throw new InvalidOperationException($"Invalid BatchCount for {nameof(YoloClsFP32SingleExtractor)}: BatchCount={BatchCount}");
    }

    public override YoloCls Extract()
    {
        return GetOutput();
    }
}
