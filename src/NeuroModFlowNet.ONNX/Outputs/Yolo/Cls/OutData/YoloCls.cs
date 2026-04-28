using System.Numerics.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Class for a single YOLO Classification result.
///     Contains the top class index and its confidence score.
/// <br/>
/// RU: Класс для результата классификации (YOLO Cls).
///     Содержит индекс лучшего класса и значение уверенности.
/// </summary>
public sealed class YoloCls : IYoloCls, IBatchedResult
{
    /// <summary> EN: Class index with the highest score. RU: Индекс класса с наивысшей оценкой. </summary>
    public int ClassId { get; }
    /// <summary> EN: Confidence score (0..1). RU: Степень уверенности (0..1). </summary>
    public float Score { get; }
    /// <summary> EN: All class scores (softmax). RU: Все оценки классов (softmax). </summary>
    public float[] Scores { get; }

    public YoloCls(int classId, float score, float[] scores)
    {
        ClassId = classId;
        Score = score;
        Scores = scores;
    }

    public YoloCls(int classId, Half score, Half[] scores)
    {
        ClassId = classId;
        Score = (float)score;
        Scores = GC.AllocateUninitializedArray<float>(scores.Length);
        TensorPrimitives.ConvertToSingle(scores, Scores);
    }

    public int BatchCount => 1;

    public override string ToString() =>
        $"[Cls] Class: {ClassId,4} | Score: {Score,5:P0} | TopK available: {Scores.Length}";
}

public readonly struct YoloClsTopKItem_FP32
{
    public readonly int ClassId;
    public readonly float Score;

    public YoloClsTopKItem_FP32(int classId, float score)
    {
        ClassId = classId;
        Score = score;
    }
}
