using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal sealed class YoloDetectionBatchResultMapCoordinatesExecutor<TPayload, TMapper> : IMapCoordinatesExecutor
    where TMapper : struct, ICoordinatePayloadMapper<TPayload>
{
    public object Map(object input, ICoordinateBackTransform transform)
    {
        YoloDetectionBatchResult<TPayload> source = (YoloDetectionBatchResult<TPayload>)input;
        var mapped = new TPayload[source.Detections.Count];
        var mapper = default(TMapper);

        for(int index = 0; index < mapped.Length; index++)
            mapped[index] = mapper.Map(source.Detections[index], transform);

        return new YoloDetectionBatchResult<TPayload>(mapped);
    }
}
