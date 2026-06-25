using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Maps crop-local coordinates back to the source image by adding the crop origin.
/// </summary>
public sealed record CropCoordinateBackTransform(float OffsetX, float OffsetY) : ICoordinateBackTransform
{
    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        mappedPoint = new Vector2(point.X + OffsetX, point.Y + OffsetY);
        return true;
    }
}

