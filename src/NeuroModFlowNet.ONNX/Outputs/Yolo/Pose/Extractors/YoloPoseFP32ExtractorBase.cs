using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Pose extractors specialized for FP32.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO Pose, специализированный для FP32.
/// </summary>
public abstract class YoloPoseFP32ExtractorBase<TOut> : YoloPoseExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var elementType = Model.GetOutputElementType(Model.PrimaryOutputName);
        if(elementType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {elementType}, but FP32 extractor requires Float.");
    }


}
