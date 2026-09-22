namespace NeuroModFlowNet.CV.Tracking;

public interface ITrackerDebugSnapshot
{
    int ConfirmedTrackCount { get; }
    int NextId { get; }
    IReadOnlyList<TrackedObject> ActiveTracks { get; }
}
