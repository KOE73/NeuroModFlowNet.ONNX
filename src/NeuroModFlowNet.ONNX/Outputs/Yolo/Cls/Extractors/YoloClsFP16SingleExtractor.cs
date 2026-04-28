namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Classification Single result extractor for FP16.
/// <br/>
/// RU: Экстрактор одного результата классификации YOLO для FP16.
/// </summary>
public class YoloClsFP16SingleExtractor : YoloClsFP16ExtractorBase<YoloCls>
{
    protected override void Check()
    {
        base.Check();
        if(BatchCount != 1)
            throw new InvalidOperationException($"Invalid BatchCount for {nameof(YoloClsFP16SingleExtractor)}: BatchCount={BatchCount}");
    }

    public override YoloCls Extract()
    {
        return GetOutput();
    }
}
