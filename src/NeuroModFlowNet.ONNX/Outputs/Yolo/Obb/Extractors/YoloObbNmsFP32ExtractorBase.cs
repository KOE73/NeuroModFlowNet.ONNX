using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO OBB NMS extractors specialized for FP32.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO OBB NMS, специализированный для FP32.
/// </summary>
public abstract class YoloObbNmsFP32ExtractorBase<TOut> : YoloObbNmsExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var meta = Model.Session.OutputMetadata[Model.PrimaryOutputName];
        if(meta.ElementDataType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP32 extractor requires Float.");
    }

    public IDetectionResult<YoloObb_FP32_XYWHSCA> GetOutput()
    {
        var data = Model.GetTensorDataAsSpan<float>();
        var allDetections = MemoryMarshal.Cast<float, YoloObb_FP32_XYWHSCA>(data);

        var result = BatchedResultPooledFactory.Create<YoloObb_FP32_XYWHSCA>(BatchCount, BatchCount * ItemCount);

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

    public IDetectionResult<YoloObb> GetOutputStd()
    {
        var data = Model.GetTensorDataAsSpan<float>();
        var allDetections = MemoryMarshal.Cast<float, YoloObb_FP32_XYWHSCA>(data);

        var result = new BatchedResultPooled<YoloObb>(BatchCount, BatchCount * ItemCount);

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
