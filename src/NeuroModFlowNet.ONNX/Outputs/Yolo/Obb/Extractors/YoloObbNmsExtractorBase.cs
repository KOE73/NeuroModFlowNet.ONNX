namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO OBB NMS extractors with data type validation.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO OBB NMS с проверкой типа данных.
/// </summary>
public abstract class YoloObbNmsExtractorBase<TOut> : ResultExtractorBase<TOut>, IExtractorThreshold
    where TOut : IBatchedResult
{
    protected override void Init()
    {
        var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        BatchCount = (int)shape[0];
        ItemCount = (int)shape[1];
    }

    public int BatchCount { get; private set; }
    public int ItemCount { get; private set; }

    public float Threshold { get; set; } = 0.5f;
}
