using System.Numerics;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly struct YoloBoxCoordinatePayloadMapper : ICoordinatePayloadMapper<YoloBox>
{
    public YoloBox Map(YoloBox payload, ICoordinateBackTransform transform)
    {
        if(!transform.TryMapBackward(new Vector2(payload.X, payload.Y), out Vector2 firstPoint) ||
           !transform.TryMapBackward(new Vector2(payload.W, payload.H), out Vector2 secondPoint))
        {
            throw new InvalidOperationException("Coordinate transform failed.");
        }

        payload.X = firstPoint.X;
        payload.Y = firstPoint.Y;
        payload.W = secondPoint.X;
        payload.H = secondPoint.Y;
        return payload;
    }
}
