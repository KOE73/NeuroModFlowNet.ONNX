using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Single VM-request YOLO Classification result.
/// </summary>
public sealed class YoloClassificationFrameResult
{
    public YoloClassificationFrameResult(int classId, float score, float[] scores)
    {
        ClassId = classId;
        Score = score;
        Scores = scores ?? throw new ArgumentNullException(nameof(scores));
    }

    public int ClassId { get; }

    public float Score { get; }

    public float[] Scores { get; }

    public static YoloClassificationFrameResult From(YoloCls result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return new YoloClassificationFrameResult(result.ClassId, result.Score, result.Scores);
    }
}
