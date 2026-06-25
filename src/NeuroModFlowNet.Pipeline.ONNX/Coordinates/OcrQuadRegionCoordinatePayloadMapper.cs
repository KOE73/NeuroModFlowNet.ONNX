using System.Numerics;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly struct OcrQuadRegionCoordinatePayloadMapper : ICoordinatePayloadMapper<OcrQuadRegion>
{
    public OcrQuadRegion Map(OcrQuadRegion payload, ICoordinateBackTransform transform)
    {
        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        Span<Point2f> mappedPoints = stackalloc Point2f[4];

        payload.CopyTo(sourcePoints);
        MapPoints(sourcePoints, mappedPoints, transform);
        return OcrQuadRegion.FromPoints(mappedPoints);
    }

    internal static void MapPoints(
        ReadOnlySpan<Point2f> sourcePoints,
        Span<Point2f> mappedPoints,
        ICoordinateBackTransform transform)
    {
        if(mappedPoints.Length < sourcePoints.Length)
            throw new ArgumentException("Destination span is shorter than source span.", nameof(mappedPoints));

        for(int index = 0; index < sourcePoints.Length; index++)
        {
            Point2f sourcePoint = sourcePoints[index];
            if(!transform.TryMapBackward(new Vector2(sourcePoint.X, sourcePoint.Y), out Vector2 mappedPoint))
                throw new InvalidOperationException("Coordinate transform failed.");

            mappedPoints[index] = new Point2f(mappedPoint.X, mappedPoint.Y);
        }
    }
}
