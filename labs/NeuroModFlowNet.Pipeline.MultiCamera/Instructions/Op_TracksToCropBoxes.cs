using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Selects tracks that are worth reading (confirmed and seen in this frame, like the old pipeline's
/// <c>AbortWhen: !IsConfirmed || MissedFrames &gt; 0</c>) and emits their boxes as <c>YoloObb[]</c> crop regions plus the
/// parallel <c>int[]</c> of track ids. At most <c>maxCount</c> regions, lowest track ids first.
///
/// RU: Отбирает треки, которые стоит читать (подтверждённые и видимые в этом кадре, как
/// <c>AbortWhen: !IsConfirmed || MissedFrames &gt; 0</c> в старом пайплайне), и отдаёт их боксы как регионы вырезки
/// <c>YoloObb[]</c> плюс параллельный <c>int[]</c> с id треков. Не более <c>maxCount</c> регионов, младшие id первыми.
/// </summary>
internal sealed class Op_TracksToCropBoxes : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly string trackIdsOutputKey;
    readonly int maxCount;

    public Op_TracksToCropBoxes(string inputKey, string outputKey, string trackIdsOutputKey, int maxCount)
        : base(OpDescriptor.Create(
            "Op_TracksToCropBoxes",
            "op.tracks.toCropBoxes",
            [VarRequirement.Read<TrackedObject[]>(inputKey)],
            [VarRequirement.Write<YoloObb[]>(outputKey), VarRequirement.Write<int[]>(trackIdsOutputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(trackIdsOutputKey);
        if(maxCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCount), "Max count must be positive.");

        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.trackIdsOutputKey = trackIdsOutputKey;
        this.maxCount = maxCount;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        TrackedObject[] tracks = context.Get<TrackedObject[]>(inputKey);
        var boxes = new List<YoloObb>(Math.Min(maxCount, tracks.Length));
        var trackIds = new List<int>(boxes.Capacity);

        foreach(TrackedObject track in tracks
            .Where(static track => track.IsConfirmed && track.MissedFrames == 0)
            .OrderBy(static track => track.ConfirmedTrackId))
        {
            if(boxes.Count >= maxCount)
                break;

            boxes.Add(new YoloObb
            {
                X = track.X,
                Y = track.Y,
                W = track.W,
                H = track.H,
                Angle = track.Angle,
                Score = track.Score,
                Class = track.ClassId
            });
            trackIds.Add(track.ConfirmedTrackId);
        }

        context.Set(outputKey, boxes.ToArray());
        context.Set(trackIdsOutputKey, trackIds.ToArray());
        return ValueTask.FromResult(OpResult.Continue);
    }
}
