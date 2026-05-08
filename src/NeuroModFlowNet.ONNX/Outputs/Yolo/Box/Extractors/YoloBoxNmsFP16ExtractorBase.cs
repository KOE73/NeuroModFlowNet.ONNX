using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Box NMS extractors specialized for FP16.
/// <br/>
/// RU: Базовый класс для экстракторов YOLO Box NMS, специализированный для FP16.
/// </summary>
public abstract class YoloBoxNmsFP16ExtractorBase<TOut> : YoloBoxNmsExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        var meta = Model.Session.OutputMetadata[Model.PrimaryOutputName];
        if(meta.ElementDataType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP16 extractor requires Float16 (Half).");
    }




    public IDetectionResult<YoloBox_FP16_XYWHSC> GetOutput()
    {
        var data = Model.GetTensorDataAsSpan<Float16>();
        var allDetections = MemoryMarshal.Cast<Float16, YoloBox_FP16_XYWHSC>(data);

        var result = BatchedResultPooledFactory.Create<YoloBox_FP16_XYWHSC>(BatchCount, BatchCount * ItemCount);

        Half threshold = (Half)Threshold;

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            var batchSpan = allDetections.Slice(batch * ItemCount, ItemCount);

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= threshold)
                    result.Add(batchSpan[itemIndex]);
        }
        return result;
    }

    public IDetectionResult<YoloBox> GetOutputStd()
    {
        var data = Model.GetTensorDataAsSpan<Float16>();
        var allDetections = MemoryMarshal.Cast<Float16, YoloBox_FP16_XYWHSC>(data);

        var result = BatchedResultPooledFactory.Create<YoloBox>(BatchCount, BatchCount * ItemCount);
        Half threshold = (Half)Threshold;

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            var batchSpan = allDetections.Slice(batch * ItemCount, ItemCount);

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= threshold)
                    result.Add(batchSpan[itemIndex].AsStd());
        }
        return result;
    }
}
