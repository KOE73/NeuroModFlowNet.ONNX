using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Box NMS extractors specialized for FP32.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO Box NMS, специализированный для FP32.
/// </summary>
public abstract class YoloBoxNmsFP32ExtractorBase<TOut> : YoloBoxNmsExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var elementType = Model.GetOutputElementType(Model.PrimaryOutputName);
        if(elementType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {elementType}, but FP32 extractor requires Float.");
    }

    public IDetectionResult<YoloBox_FP32_XYWHSC> GetOutput(IOnnxModelOutputs outputs)
    {
        var data = outputs.GetTensorDataAsSpan<float>();
        var allDetections = MemoryMarshal.Cast<float, YoloBox_FP32_XYWHSC>(data);

        var result = BatchedResultPooledFactory.Create<YoloBox_FP32_XYWHSC>(BatchCount, BatchCount * ItemCount);

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            var batchSpan = allDetections.Slice(batch * ItemCount, ItemCount);

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= Threshold)
                    result.Add(batchSpan[itemIndex]);
        }
        return result;
    }

    public IDetectionResult<YoloBox> GetOutputStd(IOnnxModelOutputs outputs)
    {
        var data = outputs.GetTensorDataAsSpan<float>();
        var allDetections = MemoryMarshal.Cast<float, YoloBox_FP32_XYWHSC>(data);

        var result = BatchedResultPooledFactory.Create<YoloBox>(BatchCount, BatchCount * ItemCount);

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            var batchSpan = allDetections.Slice(batch * ItemCount, ItemCount);

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= Threshold)
                    result.Add(batchSpan[itemIndex].AsStd());
        }
        return result;
    }
}
