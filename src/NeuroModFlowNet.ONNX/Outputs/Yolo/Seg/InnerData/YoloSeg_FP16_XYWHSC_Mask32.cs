using System.Runtime.CompilerServices;
using System.Text;

namespace NeuroModFlowNet.ONNX;

[InlineArray(Length)]
public struct InlineArray_FP16_Mask_Count32
{
    public const int Length = 32;

    private Half _element0;

    //public override string ToString() => $"[{string.Join(';', this)}]";
}

/// <summary>
/// EN: Structure for a single YOLO Segmentation detection result (Half version).
/// RU: Результат детекции сегментации (Half версия).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct YoloSeg_FP16_XYWHSC_Mask32 : IYoloSeg_XYWHSC_Mask
{
    public readonly Half X;
    public readonly Half Y;
    public readonly Half W;
    public readonly Half H;
    public readonly Half Score;
    public readonly Half ClassId;
    public readonly InlineArray_FP16_Mask_Count32 MaskCoefficients;

    public YoloSeg_FP16_XYWHSC_Mask32(Half x, Half y, Half w, Half h, Half score, Half classId, InlineArray_FP16_Mask_Count32 maskCoefs)
    {
        X = x; Y = y; W = w; H = h;
        Score = score; ClassId = classId;
        MaskCoefficients = maskCoefs;
    }


    readonly float IYoloSeg_XYWHSC_Mask.GetMaskCoefficient(int index) => (float)MaskCoefficients[index];

    public bool IsValid(float threshold) => (float)Score >= threshold;

    public override unsafe string ToString()
    {
        StringBuilder sb = new();

        for(int i=0;i< InlineArray_FP16_Mask_Count32.Length;i++)
            sb.Append($"{(float)MaskCoefficients[i],6:F3};");

        return $"[SegHalf] X,Y={(float)X:F1}-{(float)Y:F1} WxH={(float)W:F1}x{(float)H:F1} Score:{(float)Score,5:F4} | Class:{ClassId,3}  [{sb}]";
    }
}
