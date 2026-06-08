using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Per-transaction graph of coordinate transforms.
/// </summary>
/// <remarks>
/// The VM timeline stays linear, but images and regions inside one transaction can form several coordinate spaces.
/// Keeping this as a graph lets one source frame produce detector inputs, OCR crops, and rectified regions without
/// forcing every transform into one rigid metadata type.
/// </remarks>
public sealed class CoordinateTransformGraph
{
    readonly List<ICoordinateTransform> transforms = [];

    public void Add(ICoordinateTransform transform)
    {
        ArgumentNullException.ThrowIfNull(transform);
        transforms.Add(transform);
    }

    public bool TryMapBackward(
        CoordinateSpaceId fromSpace,
        CoordinateSpaceId toSpace,
        ReadOnlySpan<Vector2> fromPoints,
        Span<Vector2> toPoints)
    {
        if(fromSpace == toSpace)
        {
            fromPoints.CopyTo(toPoints);
            return true;
        }

        List<ICoordinateTransform>? path = FindBackwardPath(fromSpace, toSpace);
        if(path is null)
            return false;

        Vector2[] current = fromPoints.ToArray();
        Vector2[] next = new Vector2[current.Length];

        foreach(ICoordinateTransform transform in path)
        {
            if(!transform.TryMapBackward(current, next))
                return false;

            (current, next) = (next, current);
        }

        current.AsSpan().CopyTo(toPoints);
        return true;
    }

    List<ICoordinateTransform>? FindBackwardPath(CoordinateSpaceId fromSpace, CoordinateSpaceId toSpace)
    {
        var visited = new HashSet<CoordinateSpaceId> { fromSpace };
        var queue = new Queue<(CoordinateSpaceId Space, List<ICoordinateTransform> Path)>();
        queue.Enqueue((fromSpace, []));

        while(queue.Count > 0)
        {
            (CoordinateSpaceId space, List<ICoordinateTransform> path) = queue.Dequeue();

            foreach(ICoordinateTransform transform in transforms)
            {
                if(transform.TargetSpace != space || !visited.Add(transform.SourceSpace))
                    continue;

                List<ICoordinateTransform> nextPath = [.. path, transform];
                if(transform.SourceSpace == toSpace)
                    return nextPath;

                queue.Enqueue((transform.SourceSpace, nextPath));
            }
        }

        return null;
    }
}

