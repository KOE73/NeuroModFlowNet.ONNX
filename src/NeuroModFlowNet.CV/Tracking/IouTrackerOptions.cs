namespace NeuroModFlowNet.CV.Tracking;

public readonly record struct IouTrackerOptions(
    float IouThreshold = 0.5f,
    int MinAge = 15,
    int MaxMissedFrames = 10,
    int MaxTrailLength = 50,
    float StartZoneContainmentThreshold = 1f)
{
    public void Validate()
    {
        if(IouThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(IouThreshold), "IoU threshold must be in [0, 1].");

        if(MinAge < 0)
            throw new ArgumentOutOfRangeException(nameof(MinAge), "MinAge must be non-negative.");

        if(MaxMissedFrames < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxMissedFrames), "MaxMissedFrames must be non-negative.");

        if(MaxTrailLength < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxTrailLength), "MaxTrailLength must be non-negative.");

        if(StartZoneContainmentThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(StartZoneContainmentThreshold), "Start zone threshold must be in [0, 1].");
    }
}
