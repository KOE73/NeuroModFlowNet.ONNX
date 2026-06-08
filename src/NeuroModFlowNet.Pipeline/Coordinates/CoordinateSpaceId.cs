namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Name of a coordinate space produced or consumed by transaction payloads.
/// </summary>
public readonly record struct CoordinateSpaceId(string Value)
{
    public override string ToString() => Value;
}

