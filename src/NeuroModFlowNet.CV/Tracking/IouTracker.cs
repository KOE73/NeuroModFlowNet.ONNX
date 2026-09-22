namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Simple deterministic IoU tracker ported as neutral CPU geometry.
/// </summary>
public sealed class IouTracker : ITracker, ITrackerDebugSnapshot
{
    readonly IouTrackerOptions options;
    readonly List<TrackState> tracks = [];
    int nextId;
    int confirmedTrackCount;

    public IouTracker(IouTrackerOptions options)
    {
        options.Validate();
        this.options = options;
    }

    public int ConfirmedTrackCount => confirmedTrackCount;

    public int NextId => nextId;

    public IReadOnlyList<TrackedObject> ActiveTracks => Snapshot();

    public IReadOnlyList<TrackedObject> Process(
        IReadOnlyList<TrackDetection> detections,
        in FrameContext frame,
        in TrackerFrameOptions frameOptions)
    {
        ArgumentNullException.ThrowIfNull(detections);

        bool[] matched = new bool[detections.Count];
        List<TrackState> nextTracks = new(tracks.Count + detections.Count);

        foreach(TrackState track in tracks)
        {
            float bestIoU = 0;
            int bestIndex = -1;

            for(int index = 0; index < detections.Count; index++)
            {
                if(matched[index])
                    continue;

                TrackDetection detection = detections[index];
                if(track.ClassId != detection.ClassId)
                    continue;

                float iou = TrackRect.IoU(track.Bounds, detection.AxisAlignedBounds);
                if(iou > bestIoU)
                {
                    bestIoU = iou;
                    bestIndex = index;
                }
            }

            if(bestIoU > options.IouThreshold && bestIndex >= 0)
                UpdateMatchedTrack(track, detections[bestIndex], bestIndex);
            else
                MarkMissed(track);

            if(track.MissedFrames <= options.MaxMissedFrames)
                nextTracks.Add(track);
        }

        for(int index = 0; index < detections.Count; index++)
        {
            if(matched[index])
                continue;

            TrackDetection detection = detections[index];
            if(!PassesStartZone(detection, frameOptions.StartZone))
                continue;

            TrackState track = TrackState.Create(++nextId, detection);
            ConfirmIfReady(track);
            nextTracks.Add(track);
        }

        tracks.Clear();
        tracks.AddRange(nextTracks);
        return Snapshot();

        void UpdateMatchedTrack(TrackState track, TrackDetection detection, int matchedDetectionIndex)
        {
            track.X = detection.X;
            track.Y = detection.Y;
            track.W = detection.W;
            track.H = detection.H;
            track.Angle = detection.Angle;
            track.Score = detection.Score;
            track.SourceDetectionIndex = detection.SourceIndex;
            track.MissedFrames = 0;
            track.Age++;
            track.Path.Add(new TrackPoint(detection.X, detection.Y));

            if(track.Path.Count > options.MaxTrailLength)
                track.Path.RemoveAt(0);

            ConfirmIfReady(track);

            matched[matchedDetectionIndex] = true;
        }

        void ConfirmIfReady(TrackState track)
        {
            if(track.IsConfirmed || track.Age < options.MinAge)
                return;

            track.IsConfirmed = true;
            track.ConfirmedTrackId = ++confirmedTrackCount;
        }
    }

    bool PassesStartZone(TrackDetection detection, TrackRect? startZone)
    {
        if(startZone is null)
            return true;

        float containment = TrackRect.ContainmentRatio(startZone.Value, detection.AxisAlignedBounds);
        return containment >= options.StartZoneContainmentThreshold;
    }

    static void MarkMissed(TrackState track)
    {
        track.X = 0;
        track.Y = 0;
        track.W = 0;
        track.H = 0;
        track.Angle = 0;
        track.Score = 0;
        track.SourceDetectionIndex = -1;
        track.MissedFrames++;
    }

    TrackedObject[] Snapshot()
    {
        var snapshot = new TrackedObject[tracks.Count];
        for(int index = 0; index < snapshot.Length; index++)
            snapshot[index] = tracks[index].ToTrackedObject();

        return snapshot;
    }

    sealed class TrackState
    {
        public int TrackId { get; init; }
        public int ConfirmedTrackId { get; set; } = -1;
        public bool IsConfirmed { get; set; }
        public int Age { get; set; }
        public int MissedFrames { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float W { get; set; }
        public float H { get; set; }
        public float Angle { get; set; }
        public int ClassId { get; init; }
        public float Score { get; set; }
        public int SourceDetectionIndex { get; set; }
        public List<TrackPoint> Path { get; } = [];

        public TrackRect Bounds => TrackRect.FromCenter(X, Y, W, H);

        public static TrackState Create(int trackId, TrackDetection detection)
        {
            var state = new TrackState
            {
                TrackId = trackId,
                Age = 1,
                X = detection.X,
                Y = detection.Y,
                W = detection.W,
                H = detection.H,
                Angle = detection.Angle,
                ClassId = detection.ClassId,
                Score = detection.Score,
                SourceDetectionIndex = detection.SourceIndex
            };
            state.Path.Add(new TrackPoint(detection.X, detection.Y));
            return state;
        }

        public TrackedObject ToTrackedObject() =>
            new(
                TrackId,
                ConfirmedTrackId,
                IsConfirmed,
                Age,
                MissedFrames,
                X,
                Y,
                W,
                H,
                Angle,
                ClassId,
                Score,
                SourceDetectionIndex,
                Path.ToArray());
    }
}
