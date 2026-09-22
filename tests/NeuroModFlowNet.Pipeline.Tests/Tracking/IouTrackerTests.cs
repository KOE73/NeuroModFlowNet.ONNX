using NeuroModFlowNet.CV.Tracking;

namespace NeuroModFlowNet.Pipeline.Tests.Tracking;

public sealed class IouTrackerTests
{
    [Fact]
    public void Process_CreatesTrackOnlyInsideStartZone()
    {
        var tracker = new IouTracker(new IouTrackerOptions(
            IouThreshold: 0.5f,
            MinAge: 1,
            MaxMissedFrames: 2,
            MaxTrailLength: 10,
            StartZoneContainmentThreshold: 1));
        var frame = new FrameContext(0, DateTimeOffset.UnixEpoch, 0);
        var options = new TrackerFrameOptions(new TrackRect(0, 0, 100, 100));

        IReadOnlyList<TrackedObject> outside = tracker.Process(
            [Detection(140, 140, sourceIndex: 0)],
            frame,
            options);
        IReadOnlyList<TrackedObject> inside = tracker.Process(
            [Detection(50, 50, sourceIndex: 0)],
            frame with { RunId = 1 },
            options);

        Assert.Empty(outside);
        TrackedObject track = Assert.Single(inside);
        Assert.Equal(1, track.TrackId);
        Assert.Equal(0, track.SourceDetectionIndex);
    }

    [Fact]
    public void Process_ConfirmsTrackAfterMinAge()
    {
        var tracker = new IouTracker(new IouTrackerOptions(
            IouThreshold: 0.2f,
            MinAge: 3,
            MaxMissedFrames: 2,
            MaxTrailLength: 10));
        var options = new TrackerFrameOptions();

        IReadOnlyList<TrackedObject> first = tracker.Process([Detection(10, 10)], new FrameContext(0, DateTimeOffset.UnixEpoch, 0), options);
        IReadOnlyList<TrackedObject> second = tracker.Process([Detection(11, 10)], new FrameContext(1, DateTimeOffset.UnixEpoch, 1), options);
        IReadOnlyList<TrackedObject> third = tracker.Process([Detection(12, 10)], new FrameContext(2, DateTimeOffset.UnixEpoch, 2), options);

        Assert.False(Assert.Single(first).IsConfirmed);
        Assert.False(Assert.Single(second).IsConfirmed);
        TrackedObject confirmed = Assert.Single(third);
        Assert.True(confirmed.IsConfirmed);
        Assert.Equal(1, confirmed.ConfirmedTrackId);
    }

    [Fact]
    public void Process_DropsTrackAfterMaxMissedFrames()
    {
        var tracker = new IouTracker(new IouTrackerOptions(
            IouThreshold: 0.5f,
            MinAge: 1,
            MaxMissedFrames: 1,
            MaxTrailLength: 10));
        var options = new TrackerFrameOptions();

        IReadOnlyList<TrackedObject> first = tracker.Process([Detection(10, 10)], new FrameContext(0, DateTimeOffset.UnixEpoch, 0), options);
        IReadOnlyList<TrackedObject> missedOnce = tracker.Process([], new FrameContext(1, DateTimeOffset.UnixEpoch, 1), options);
        IReadOnlyList<TrackedObject> missedTwice = tracker.Process([], new FrameContext(2, DateTimeOffset.UnixEpoch, 2), options);

        Assert.Single(first);
        TrackedObject missed = Assert.Single(missedOnce);
        Assert.Equal(1, missed.MissedFrames);
        Assert.Equal(-1, missed.SourceDetectionIndex);
        Assert.Empty(missedTwice);
    }

    [Fact]
    public void Process_MatchesOnlyWithinSameClass()
    {
        var tracker = new IouTracker(new IouTrackerOptions(
            IouThreshold: 0.2f,
            MinAge: 1,
            MaxMissedFrames: 2,
            MaxTrailLength: 10));
        var options = new TrackerFrameOptions();

        tracker.Process([Detection(10, 10, classId: 1)], new FrameContext(0, DateTimeOffset.UnixEpoch, 0), options);
        IReadOnlyList<TrackedObject> tracks = tracker.Process([Detection(11, 10, classId: 2)], new FrameContext(1, DateTimeOffset.UnixEpoch, 1), options);

        Assert.Equal(2, tracks.Count);
        Assert.Contains(tracks, track => track.ClassId == 1 && track.MissedFrames == 1);
        Assert.Contains(tracks, track => track.ClassId == 2 && track.MissedFrames == 0);
    }

    [Fact]
    public void Process_IsDeterministicForSameSequence()
    {
        static int[] Run()
        {
            var tracker = new IouTracker(new IouTrackerOptions(
                IouThreshold: 0.2f,
                MinAge: 2,
                MaxMissedFrames: 2,
                MaxTrailLength: 10));
            var options = new TrackerFrameOptions();

            tracker.Process([Detection(10, 10), Detection(100, 100, sourceIndex: 1)], new FrameContext(0, DateTimeOffset.UnixEpoch, 0), options);
            IReadOnlyList<TrackedObject> tracks = tracker.Process([Detection(11, 10), Detection(102, 100, sourceIndex: 1)], new FrameContext(1, DateTimeOffset.UnixEpoch, 1), options);
            return tracks.Select(track => track.TrackId).ToArray();
        }

        Assert.Equal(Run(), Run());
    }

    static TrackDetection Detection(
        float x,
        float y,
        int classId = 0,
        int sourceIndex = 0) =>
        new(x, y, 20, 20, 0, classId, 0.9f, sourceIndex);
}
