namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Neutral CPU detection consumed by trackers.
/// </summary>
public readonly record struct TrackDetection(
    float X,
    float Y,
    float W,
    float H,
    float Angle,
    int ClassId,
    float Score,
    int SourceIndex)
{
    public TrackRect AxisAlignedBounds => TrackRect.FromCenter(X, Y, W, H);
}
