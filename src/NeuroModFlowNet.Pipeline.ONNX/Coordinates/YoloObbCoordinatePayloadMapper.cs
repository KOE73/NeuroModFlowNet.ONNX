using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly struct YoloObbCoordinatePayloadMapper : ICoordinatePayloadMapper<YoloObb>
{
    public YoloObb Map(
        YoloObb payload,
        ICoordinateBackTransform transform,
        CoordinateMappingShapePolicy shapePolicy)
    {
        EnsureSupportedShapePolicy(shapePolicy);

        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        Span<Point2f> mappedPoints = stackalloc Point2f[4];

        var region = new OcrObbRegion(
            payload.X,
            payload.Y,
            Math.Max(2, payload.W),
            Math.Max(2, payload.H),
            payload.Angle);

        region.GetPoints(sourcePoints);
        OcrQuadRegionCoordinatePayloadMapper.MapPoints(sourcePoints, mappedPoints, transform);

        Point2f center = GetCenter(mappedPoints);
        float width = Distance(mappedPoints[0], mappedPoints[1]);
        float height = Distance(mappedPoints[1], mappedPoints[2]);
        float angle = MathF.Atan2(
            mappedPoints[1].Y - mappedPoints[0].Y,
            mappedPoints[1].X - mappedPoints[0].X);

        payload.X = center.X;
        payload.Y = center.Y;
        payload.W = width;
        payload.H = height;
        payload.Angle = angle;
        return payload;
    }

    static void EnsureSupportedShapePolicy(CoordinateMappingShapePolicy shapePolicy)
    {
        if(shapePolicy is CoordinateMappingShapePolicy.PreserveShape or CoordinateMappingShapePolicy.BoundingOBB)
            return;

        throw new NotSupportedException($"Shape policy '{shapePolicy}' is not supported for {nameof(YoloObb)} payloads yet.");
    }

    static Point2f GetCenter(ReadOnlySpan<Point2f> points) =>
        new(
            (points[0].X + points[1].X + points[2].X + points[3].X) * 0.25f,
            (points[0].Y + points[1].Y + points[2].Y + points[3].Y) * 0.25f);

    static float Distance(Point2f first, Point2f second)
    {
        float dx = second.X - first.X;
        float dy = second.Y - first.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
