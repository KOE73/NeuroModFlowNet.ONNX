using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal sealed class ArrayMapCoordinatesExecutor<TPayload, TMapper> : IMapCoordinatesExecutor
    where TMapper : struct, ICoordinatePayloadMapper<TPayload>
{
    public object Map(
        object input,
        ICoordinateBackTransform transform,
        CoordinateMappingShapePolicy shapePolicy)
    {
        TPayload[] source = (TPayload[])input;
        var mapped = new TPayload[source.Length];
        var mapper = default(TMapper);

        for(int index = 0; index < source.Length; index++)
            mapped[index] = mapper.Map(source[index], transform, shapePolicy);

        return mapped;
    }
}
