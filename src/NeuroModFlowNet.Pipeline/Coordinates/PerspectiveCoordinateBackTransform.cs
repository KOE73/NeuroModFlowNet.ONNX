using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

public readonly record struct PerspectiveCoordinateBackTransform(
    float M00,
    float M01,
    float M02,
    float M10,
    float M11,
    float M12,
    float M20,
    float M21,
    float M22) : ICoordinateBackTransform
{
    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        float denominator = M20 * point.X + M21 * point.Y + M22;
        if(MathF.Abs(denominator) <= float.Epsilon)
        {
            mappedPoint = default;
            return false;
        }

        mappedPoint = new Vector2(
            (M00 * point.X + M01 * point.Y + M02) / denominator,
            (M10 * point.X + M11 * point.Y + M12) / denominator);
        return true;
    }
}
