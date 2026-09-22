using System.Numerics;

namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Axis-aligned rectangle used by tracker math and per-source zones.
/// </summary>
public readonly record struct TrackRect(float X1, float Y1, float X2, float Y2)
{
    public static TrackRect Empty { get; } = new(0, 0, 0, 0);

    public float Left => MathF.Min(X1, X2);
    public float Top => MathF.Min(Y1, Y2);
    public float Right => MathF.Max(X1, X2);
    public float Bottom => MathF.Max(Y1, Y2);
    public float Width => Right - Left;
    public float Height => Bottom - Top;
    public float Area => MathF.Max(0, Width) * MathF.Max(0, Height);
    public bool IsEmpty => Width <= 0 || Height <= 0;
    public TrackPoint Center => new((Left + Right) * 0.5f, (Top + Bottom) * 0.5f);

    public static TrackRect FromCenter(float centerX, float centerY, float width, float height) =>
        new(
            centerX - (width * 0.5f),
            centerY - (height * 0.5f),
            centerX + (width * 0.5f),
            centerY + (height * 0.5f));

    public static float IoU(TrackRect first, TrackRect second)
    {
        Vector2 intersectionMin = Vector2.Max(
            new Vector2(first.Left, first.Top),
            new Vector2(second.Left, second.Top));
        Vector2 intersectionMax = Vector2.Min(
            new Vector2(first.Right, first.Bottom),
            new Vector2(second.Right, second.Bottom));
        Vector2 intersectionSize = Vector2.Max(Vector2.Zero, intersectionMax - intersectionMin);

        float intersectionArea = intersectionSize.X * intersectionSize.Y;
        float unionArea = first.Area + second.Area - intersectionArea;
        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    public static float ContainmentRatio(TrackRect container, TrackRect item)
    {
        Vector2 intersectionMin = Vector2.Max(
            new Vector2(container.Left, container.Top),
            new Vector2(item.Left, item.Top));
        Vector2 intersectionMax = Vector2.Min(
            new Vector2(container.Right, container.Bottom),
            new Vector2(item.Right, item.Bottom));
        Vector2 intersectionSize = Vector2.Max(Vector2.Zero, intersectionMax - intersectionMin);

        float intersectionArea = intersectionSize.X * intersectionSize.Y;
        return item.Area > 0 ? intersectionArea / item.Area : 0;
    }
}
