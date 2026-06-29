using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

public readonly struct Rotate90CoordinateBackTransform : ICoordinateBackTransform
{
    public Rotate90CoordinateBackTransform(int sourceWidth, int sourceHeight, Rotate90Mode mode)
    {
        if(sourceWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Source width must be positive.");

        if(sourceHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceHeight), "Source height must be positive.");

        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        Mode = mode;
    }

    public int SourceWidth { get; }

    public int SourceHeight { get; }

    public Rotate90Mode Mode { get; }

    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        mappedPoint = Mode switch
        {
            Rotate90Mode.Clockwise90 => new Vector2(point.Y, SourceHeight - 1 - point.X),
            Rotate90Mode.Rotate180 => new Vector2(SourceWidth - 1 - point.X, SourceHeight - 1 - point.Y),
            Rotate90Mode.CounterClockwise90 => new Vector2(SourceWidth - 1 - point.Y, point.X),
            _ => throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Unsupported rotate mode.")
        };

        return true;
    }
}
