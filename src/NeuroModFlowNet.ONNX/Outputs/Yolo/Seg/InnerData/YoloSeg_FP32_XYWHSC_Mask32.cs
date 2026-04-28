using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX;


[InlineArray(32)]
public struct InlineArray_FP32_Mask32
{
    private float _element0;
}

/// <summary>
/// EN: Structure for a single YOLO Segmentation detection result.
///     Contains bounding box, score, class id and mask coefficients
///     (used to combine with prototype masks from the second model output).
/// <br/>
/// RU: Структура для одного результата детекции сегментации (YOLO Seg NMS).
///     Содержит бокс, оценку, класс и коэффициенты маски
///     (используются совместно с прототипами масок из второго выхода модели).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct YoloSeg_FP32_XYWHSC_Mask32 : IYoloSeg_XYWHSC_Mask
{
    public readonly float X;
    public readonly float Y;
    public readonly float W;
    public readonly float H;
    public readonly float Score;
    public readonly float ClassId;
    public readonly InlineArray_FP32_Mask32 MaskCoefficients;

    public YoloSeg_FP32_XYWHSC_Mask32(float x, float y, float w, float h, float score, float classId, InlineArray_FP32_Mask32 maskCoefs)
    {
        X = x; Y = y; W = w; H = h;
        Score = score; ClassId = classId;
        MaskCoefficients = maskCoefs;
    }


    readonly float IYoloSeg_XYWHSC_Mask.GetMaskCoefficient(int index) => MaskCoefficients[index];

    public bool IsValid(float threshold) => Score >= threshold;

    public override string ToString() =>
        $"[Seg] {X,7:F1}, {Y,7:F1} | {W,6:F1}x{H,6:F1} | Score: {Score,5:P0} | Class: {ClassId,3} ";
}
