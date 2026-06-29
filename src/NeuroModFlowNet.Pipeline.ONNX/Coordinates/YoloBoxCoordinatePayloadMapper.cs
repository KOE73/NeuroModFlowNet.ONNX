using System.Numerics;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly struct YoloBoxCoordinatePayloadMapper : ICoordinatePayloadMapper<YoloBox>
{
    public YoloBox Map(
        YoloBox payload,
        ICoordinateBackTransform transform,
        CoordinateMappingShapePolicy shapePolicy)
    {
        EnsureSupportedShapePolicy(shapePolicy);

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

    static void EnsureSupportedShapePolicy(CoordinateMappingShapePolicy shapePolicy)
    {
        if(shapePolicy is CoordinateMappingShapePolicy.PreserveShape or CoordinateMappingShapePolicy.BoundingBox)
            return;

        throw new NotSupportedException($"Shape policy '{shapePolicy}' is not supported for {nameof(YoloBox)} payloads yet.");
    }
}
