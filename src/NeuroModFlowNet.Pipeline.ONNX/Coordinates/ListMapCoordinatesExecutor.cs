using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal sealed class ListMapCoordinatesExecutor<TPayload, TMapper> : IMapCoordinatesExecutor
    where TMapper : struct, ICoordinatePayloadMapper<TPayload>
{
    public object Map(
        object input,
        ICoordinateBackTransform transform,
        CoordinateMappingShapePolicy shapePolicy)
    {
        List<TPayload> source = (List<TPayload>)input;
        var mapped = new List<TPayload>(source.Count);
        var mapper = default(TMapper);

        for(int index = 0; index < source.Count; index++)
            mapped.Add(mapper.Map(source[index], transform, shapePolicy));

        return mapped;
    }
}
