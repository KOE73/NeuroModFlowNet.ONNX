using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Classification extractors specialized for FP16.
/// <br/>
/// RU: Базовый класс для экстракторов классификации YOLO, специализированный для FP16.
/// </summary>
public abstract class YoloClsFP16ExtractorBase<TOut> : YoloClsExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var meta = Model.Session.OutputMetadata[Model.PrimaryOutputName];
        if(meta.ElementDataType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP16 extractor requires Float16.");
    }

    public YoloCls GetOutput()
    {
        var data = Model.GetTensorDataAsSpan<Half>();
        int offset = 0;

        int bestIdx = 0;
        Half bestVal = Half.MinValue;

        Half[] scores = new Half[ClassesCount];
        for(int classIndex = 0; classIndex < ClassesCount; classIndex++)
        {
            Half val = data[offset + classIndex];
            scores[classIndex] = val;
            if(val > bestVal) { bestVal = val; bestIdx = classIndex; }
        }

        return new YoloCls(bestIdx, bestVal, scores);
    }
}
