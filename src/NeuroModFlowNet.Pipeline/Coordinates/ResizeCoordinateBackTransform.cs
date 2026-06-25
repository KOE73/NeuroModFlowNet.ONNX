using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Maps resized-image coordinates back to the image size that entered the resize operation.
/// </summary>
public sealed record ResizeCoordinateBackTransform(float ScaleX, float ScaleY) : ICoordinateBackTransform
{
    public ResizeCoordinateBackTransform(
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight)
        : this(
            sourceWidth / (float)targetWidth,
            sourceHeight / (float)targetHeight)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sourceWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sourceHeight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetHeight);
    }

    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        mappedPoint = new Vector2(point.X * ScaleX, point.Y * ScaleY);
        return true;
    }
}

