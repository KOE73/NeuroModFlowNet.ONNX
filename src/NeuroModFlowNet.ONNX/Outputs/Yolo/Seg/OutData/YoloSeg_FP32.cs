namespace NeuroModFlowNet.ONNX;

public readonly struct YoloSeg_FP32
{
    public readonly float X;
    public readonly float Y;
    public readonly float W;
    public readonly float H;
    public readonly float Score;
    public readonly float ClassId;
    /// <summary> Mask coefficients for combining with prototypes </summary>
    public readonly float[] MaskCoefficients;

    public YoloSeg_FP32(float x, float y, float w, float h, float score, float classId, float[] maskCoefs)
    {
        X = x; Y = y; W = w; H = h;
        Score = score; ClassId = classId;
        MaskCoefficients = maskCoefs;
    }

    public bool IsValid(float threshold) => Score >= threshold;

    public override string ToString() =>
        $"[Seg] {X,7:F1}, {Y,7:F1} | {W,6:F1}x{H,6:F1} | Score: {Score,5:P0} | Class: {ClassId,3} | Coefs: {MaskCoefficients.Length}";
}
