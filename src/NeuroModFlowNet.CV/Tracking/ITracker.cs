namespace NeuroModFlowNet.CV.Tracking;

public interface ITracker
{
    IReadOnlyList<TrackedObject> Process(
        IReadOnlyList<TrackDetection> detections,
        in FrameContext frame,
        in TrackerFrameOptions options);
}
