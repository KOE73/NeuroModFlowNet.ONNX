using NeuroModFlowNet.CV.Tracking;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Lock-protected per-camera counters read by the dashboard thread while the worker updates them.
///
/// RU: Счётчики камеры под lock: воркер обновляет, поток дашборда читает.
/// </summary>
internal sealed class CameraStatistics
{
    const double SmoothingFactor = 0.1;
    readonly object syncRoot = new();

    long framesRead;
    long runsStarted;
    long runsRejected;
    long runsCompleted;
    long runsFailed;
    double averageRunMilliseconds;
    double averageFps;
    DateTime? lastCompletedAt;
    int lastDetectionCount;
    int lastTrackCount;
    int lastConfirmedTrackCount;
    int maxTrackId;
    string lastText = string.Empty;
    string? lastError;
    string phase = "starting";
    readonly Dictionary<string, double> instructionMilliseconds = new(StringComparer.Ordinal);

    public void MarkFrameRead()
    {
        lock(syncRoot)
            framesRead++;
    }

    public void MarkRunStarted()
    {
        lock(syncRoot)
            runsStarted++;
    }

    public void MarkRunRejected()
    {
        lock(syncRoot)
            runsRejected++;
    }

    public void SetPhase(string value)
    {
        lock(syncRoot)
            phase = value;
    }

    public void MarkRunCompleted(TimeSpan duration, int detectionCount, TrackedObject[] tracks, string? text)
    {
        lock(syncRoot)
        {
            runsCompleted++;
            averageRunMilliseconds = averageRunMilliseconds == 0
                ? duration.TotalMilliseconds
                : averageRunMilliseconds + (SmoothingFactor * (duration.TotalMilliseconds - averageRunMilliseconds));

            DateTime now = DateTime.UtcNow;
            if(lastCompletedAt is { } previous && now > previous)
            {
                double instantFps = 1.0 / (now - previous).TotalSeconds;
                averageFps = averageFps == 0 ? instantFps : averageFps + (SmoothingFactor * (instantFps - averageFps));
            }

            lastCompletedAt = now;
            lastDetectionCount = detectionCount;
            lastTrackCount = tracks.Length;
            lastConfirmedTrackCount = tracks.Count(static track => track.IsConfirmed);
            foreach(TrackedObject track in tracks)
                maxTrackId = Math.Max(maxTrackId, track.TrackId);

            if(!string.IsNullOrWhiteSpace(text))
                lastText = text;

            phase = "running";
        }
    }

    /// <summary>Exponential moving average of per-instruction time, keyed by "program/instruction".</summary>
    public void AddInstructionTimings(VmRunTrace trace)
    {
        lock(syncRoot)
        {
            foreach(OpTraceEntry entry in trace.Instructions)
            {
                string key = $"{entry.Program}/{entry.Name}";
                instructionMilliseconds[key] = instructionMilliseconds.TryGetValue(key, out double previous)
                    ? previous + (SmoothingFactor * (entry.ElapsedMilliseconds - previous))
                    : entry.ElapsedMilliseconds;
            }
        }
    }

    public IReadOnlyList<KeyValuePair<string, double>> SnapshotInstructionTimings()
    {
        lock(syncRoot)
            return instructionMilliseconds.OrderByDescending(static pair => pair.Value).ToArray();
    }

    public void MarkRunFailed(string error)
    {
        lock(syncRoot)
        {
            runsFailed++;
            lastError = error;
        }
    }

    public CameraStatisticsSnapshot Snapshot()
    {
        lock(syncRoot)
        {
            return new CameraStatisticsSnapshot(
                phase,
                framesRead,
                runsStarted,
                runsRejected,
                runsCompleted,
                runsFailed,
                averageRunMilliseconds,
                averageFps,
                lastDetectionCount,
                lastTrackCount,
                lastConfirmedTrackCount,
                maxTrackId,
                lastText,
                lastError);
        }
    }
}

internal sealed record CameraStatisticsSnapshot(
    string Phase,
    long FramesRead,
    long RunsStarted,
    long RunsRejected,
    long RunsCompleted,
    long RunsFailed,
    double AverageRunMilliseconds,
    double AverageFps,
    int LastDetectionCount,
    int LastTrackCount,
    int LastConfirmedTrackCount,
    int MaxTrackId,
    string LastText,
    string? LastError);
