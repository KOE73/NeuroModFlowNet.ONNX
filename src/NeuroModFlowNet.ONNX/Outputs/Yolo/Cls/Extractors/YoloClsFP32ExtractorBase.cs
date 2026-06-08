using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Classification extractors specialized for FP32.
/// <br/>
/// RU: Базовый класс для экстракторов классификации YOLO, специализированный для FP32.
/// </summary>
public abstract class YoloClsFP32ExtractorBase<TOut> : YoloClsExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var elementType = Model.GetOutputElementType(Model.PrimaryOutputName);
        if(elementType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {elementType}, but FP32 extractor requires Float.");
    }

    public YoloCls GetOutput(IOnnxModelOutputs outputs)
    {
        var data = outputs.GetTensorDataAsSpan<float>();
        var batchSpan = data.Slice(0, ClassesCount);

        int bestIndex = TensorPrimitives.IndexOfMax(batchSpan);
        float bestScore = batchSpan[bestIndex];

        return new YoloCls(bestIndex, bestScore, batchSpan.ToArray());
    }
}
