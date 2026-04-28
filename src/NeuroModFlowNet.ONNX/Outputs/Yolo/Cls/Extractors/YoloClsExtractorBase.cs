namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Classification extractors.
/// <br/>
/// RU: Базовый класс для экстракторов классификации YOLO.
/// </summary>
public abstract class YoloClsExtractorBase<TOut> : ResultExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Init()
    {
        var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        BatchCount = (int)shape[0];
        ClassesCount = (int)shape[1];
    }

    public int BatchCount { get; private set; }
    public int ClassesCount { get; private set; }

    public float Threshold { get; set; } = 0.5f;
}
