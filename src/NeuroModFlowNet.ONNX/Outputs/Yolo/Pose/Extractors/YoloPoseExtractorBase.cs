namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Pose extractors with shared properties and initialization.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO Pose с общими свойствами и инициализацией.
/// </summary>
public abstract class YoloPoseExtractorBase<TOut> : ResultExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Init()
    {
        var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        BatchCount = (int)shape[0];
        ItemCount = (int)shape[1];
        RowSize = (int)shape[2];

        // 6 => XYHWSC
        // 3 => XYV
        KeypointsCount = (RowSize - 6) / 3;
    }

    public int BatchCount { get; private set; }
    public int ItemCount { get; private set; }
    public int RowSize { get; private set; }
    public int KeypointsCount { get; private set; }


    public float Threshold { get; set; } = 0.5f;
}
