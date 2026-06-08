using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Maps points between two coordinate spaces.
/// </summary>
/// <remarks>
/// The contract is intentionally not a fixed resize/letterbox metadata record. Real pipelines can add rotation,
/// homography, lens correction, nonlinear warps, and custom GPU-generated maps while keeping the extractor contract
/// unchanged.
/// </remarks>
public interface ICoordinateTransform
{
    CoordinateSpaceId SourceSpace { get; }

    CoordinateSpaceId TargetSpace { get; }

    bool TryMapForward(ReadOnlySpan<Vector2> sourcePoints, Span<Vector2> targetPoints);

    bool TryMapBackward(ReadOnlySpan<Vector2> targetPoints, Span<Vector2> sourcePoints);
}

