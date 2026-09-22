namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Neutral tracking result. Domain annotations such as OCR rows belong in separate pipeline registers.
/// </summary>
public sealed record TrackedObject(
    int TrackId,
    int ConfirmedTrackId,
    bool IsConfirmed,
    int Age,
    int MissedFrames,
    float X,
    float Y,
    float W,
    float H,
    float Angle,
    int ClassId,
    float Score,
    int SourceDetectionIndex,
    IReadOnlyList<TrackPoint> Path)
{
    public TrackRect AxisAlignedBounds => TrackRect.FromCenter(X, Y, W, H);
}
