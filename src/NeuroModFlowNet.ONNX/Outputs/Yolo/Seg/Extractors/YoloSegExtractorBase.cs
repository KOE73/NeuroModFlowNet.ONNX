namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Seg extractors with shared properties and initialization.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO Seg с общими свойствами и инициализацией.
/// </summary>
/// <remarks>
///  OUT : output0 | Detection | Float  | [1x300x38]
///  OUT : output1 | Prototype | TOut   | [1x32x160x160]
/// 
/// </remarks>
public abstract class YoloSegExtractorBase<TOut> : ResultExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Init()
    {
        var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        BatchCount = (int)shape[0];
        ItemCount = (int)shape[1];
        ValuesPerDetection = (int)shape[2];

        DetectionWithMaskCoefficientsOutputName = Model.OutputNames[0];
        MaskPrototypesOutputName = Model.OutputNames[1];
        MaskPrototypesShape = Model.ModelOutputShapes[MaskPrototypesOutputName];

        MaskPrototypesCount = (int)MaskPrototypesShape[1];
        MaskPrototypesHeight = (int)MaskPrototypesShape[2];
        MaskPrototypesWidth = (int)MaskPrototypesShape[3];
        MaskPixels = MaskPrototypesHeight * MaskPrototypesWidth;
    }

    protected override void Check()
    {
        if(BatchCount != 1)
            throw new InvalidOperationException($"YOLO Seg extractors currently support only batch size 1, but model output metadata reports BatchCount={BatchCount}.");
    }

    public int BatchCount { get; private set; }
    public int ItemCount { get; private set; }
    public int ValuesPerDetection { get; private set; }

    public string DetectionWithMaskCoefficientsOutputName { get; private set; } = default!;
    public string MaskPrototypesOutputName { get; private set; } = default!;
    public long[] MaskPrototypesShape { get; private set; } = default!;

    public float Threshold { get; set; } = 0.5f;

    public int MaskPrototypesCount { get; private set; }
    public int MaskPrototypesHeight { get; private set; }
    public int MaskPrototypesWidth { get; private set; }
    public int MaskPixels { get; private set; }
}
