using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

public sealed record PadResizeCoordinateBackTransform(
    int SourceWidth,
    int SourceHeight,
    int ResizedWidth,
    int ResizedHeight,
    int OutputWidth,
    int OutputHeight,
    float Scale,
    int PadLeft,
    int PadTop,
    int PadRight,
    int PadBottom,
    int Stride,
    PadResizeMode Mode) : ICoordinateBackTransform
{
    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        mappedPoint = new Vector2(
            (point.X - PadLeft) / Scale,
            (point.Y - PadTop) / Scale);
        return true;
    }
}
