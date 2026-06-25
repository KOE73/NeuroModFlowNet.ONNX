using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Maps coordinates embedded inside one domain payload value.
/// </summary>
public interface ICoordinatePayloadMapper<TPayload>
{
    TPayload Map(TPayload payload, ICoordinateBackTransform transform);
}
