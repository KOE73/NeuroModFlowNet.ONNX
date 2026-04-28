using OpenCvSharp.Flann;
using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Represent a single keypoint (x, y, visibility).
/// RU: Представляет одну ключевую точку (x, y, видимость).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct YoloPose_FP32_Keypoint_XYV : IOutAsT<YoloPoseKeypointXYV>
{
    public readonly float X;
    public readonly float Y;
    public readonly float V;

    public YoloPose_FP32_Keypoint_XYV(float x, float y, float v) { X = x; Y = y; V = v; }

    public YoloPoseKeypointXYV AsStd() => new YoloPoseKeypointXYV(X, Y, V);

    public override string ToString() => $"({X:F1}, {Y:F1}, v={V:F2})";
}

[InlineArray(17)]
public struct YoloPose_FP32_InlineKeypoints17
{
    private YoloPose_FP32_Keypoint_XYV _keypoint0;
}

/// <summary>
/// proof-of-concept
/// EN: High-performance YOLO Pose detection structure (exactly 57 floats).
///     Layout: [X, Y, W, H, Score, ClassId, K0.X, K0.Y, K0.V, ..., K16.V]
/// <br/>
/// RU: Высокопроизводительная структура YOLO Pose (ровно 57 float).
///     Структура: [X, Y, W, H, Score, ClassId, K0.X, K0.Y, K0.V, ..., K16.V]
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct YoloPose_FP32_Size57_Keypoint17 : IOutAsT<YoloPose>
{
    // Bounding Box (6 fields)
    public readonly float X;
    public readonly float Y;
    public readonly float W;
    public readonly float H;
    public readonly float Score;
    public readonly float ClassId;

    // 17 Keypoints (17 * 3 = 51 fields)
    public readonly YoloPose_FP32_InlineKeypoints17 Keypoints;

    ///// <summary> EN: Access keypoints by index (0-16). RU: Доступ к точкам по индексу (0-16). </summary>
    //public YoloPose_single_keypoint GetKeypoint(int index) => index switch
    //{
    //    0 => K0, 1 => K1, 2 => K2, 3 => K3, 4 => K4, 5 => K5, 6 => K6, 7 => K7, 8 => K8, 
    //    9 => K9, 10 => K10, 11 => K11, 12 => K12, 13 => K13, 14 => K14, 15 => K15, 16 => K16,
    //    _ => throw new IndexOutOfRangeException()
    //};
    /// <summary> EN: Access keypoints by index (0-16). RU: Доступ к точкам по индексу (0-16). </summary>
    public YoloPose_FP32_Keypoint_XYV GetKeypoint(int index)
    {
        Debug.Assert((uint)index < 17u);
        return Keypoints[index];
    }

    public YoloPose AsStd()
    {
        var keypoints = new YoloPoseKeypointXYV[17];

        for(int i = 0; i < 17; i++)
        {
            var kp = GetKeypoint(i);
            keypoints[i] = new YoloPoseKeypointXYV(kp.X, kp.Y, kp.V);
        }

        return new YoloPose
        {
            X = X,
            Y = Y,
            W = W,
            H = H,
            Score = Score,
            ClassId = ClassId,
            Keypoints = keypoints
        };
    }

    public override string ToString() => $"[Pose] {X:F1},{Y:F1} Score:{Score:P0} Kpts:17";

}
