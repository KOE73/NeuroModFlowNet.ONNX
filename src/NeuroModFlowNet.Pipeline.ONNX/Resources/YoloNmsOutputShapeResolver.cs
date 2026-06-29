using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class YoloNmsOutputShapeResolver : IOrtValueOutputShapeResolver
{
    readonly int defaultItemCount;
    readonly int fieldCount;

    public YoloNmsOutputShapeResolver(int defaultItemCount, int fieldCount)
    {
        if(defaultItemCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(defaultItemCount), "Default item count must be positive.");

        if(fieldCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(fieldCount), "Field count must be positive.");

        this.defaultItemCount = defaultItemCount;
        this.fieldCount = fieldCount;
    }

    public long[] ResolveOutputShape(OnnxModel model, string outputName, int requestCount)
    {
        ArgumentNullException.ThrowIfNull(model);

        long[] modelShape = model.ModelOutputShapes[outputName];
        if(modelShape.Length != 3)
            throw new InvalidOperationException($"YOLO NMS output '{outputName}' must be rank 3, actual rank: {modelShape.Length}.");

        long resolvedBatch = modelShape[0] > 0 ? modelShape[0] : requestCount;
        long resolvedItems = modelShape[1] > 0 ? modelShape[1] : defaultItemCount;
        long resolvedFields = modelShape[2] > 0 ? modelShape[2] : fieldCount;

        if(resolvedBatch != requestCount)
            throw new InvalidOperationException($"YOLO NMS output batch must match request count. Output batch: {resolvedBatch}, request count: {requestCount}.");

        if(resolvedFields != fieldCount)
            throw new InvalidOperationException($"YOLO NMS output field count must be {fieldCount}, actual: {resolvedFields}.");

        return [resolvedBatch, resolvedItems, resolvedFields];
    }
}
