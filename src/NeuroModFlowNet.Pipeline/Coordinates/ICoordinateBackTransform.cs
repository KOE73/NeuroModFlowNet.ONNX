using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Maps one point from the current coordinate space back to the previous coordinate space.
/// </summary>
/// <remarks>
/// VM programs keep transforms as normal named registers. Chained mapping is expressed by multiple explicit
/// instructions so every intermediate coordinate payload can be inspected or reused.
/// </remarks>
public interface ICoordinateBackTransform
{
    bool TryMapBackward(Vector2 point, out Vector2 mappedPoint);
}

