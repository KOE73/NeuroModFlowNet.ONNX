using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal sealed class MapCoordinatesExecutor<TPayload, TMapper> : IMapCoordinatesExecutor
    where TMapper : struct, ICoordinatePayloadMapper<TPayload>
{
    public object Map(object input, ICoordinateBackTransform transform)
    {
        TPayload payload = (TPayload)input;
        return default(TMapper).Map(payload, transform)!;
    }
}
