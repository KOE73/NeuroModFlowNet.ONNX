using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Visualizer;

public static class VisualUtils
{
    // ---------- COCO 80-class palette (hue-stepped) ----------
    public static readonly Scalar[] ClassColors = Enumerable
        .Range(0, 80)
        .Select(i => HsvToScalar(i * 4.5f, 220, 200))
        .ToArray();

    public static Scalar HsvToScalar(float h, float s, float v)
    {
        using var hsv = new Mat(1, 1, MatType.CV_8UC3, new Scalar(h, s, v));
        using var bgr = new Mat();
        Cv2.CvtColor(hsv, bgr, ColorConversionCodes.HSV2BGR);
        var idx = bgr.At<Vec3b>(0, 0);
        return new Scalar(idx.Item0, idx.Item1, idx.Item2);
    }

    public static Scalar ClassColor(int classId)
    {
        int idx = Math.Abs(classId) % ClassColors.Length;
        return ClassColors[idx];
    }

    public static SKColor ClassColorSkia(int classId)
    {
        var scalar = ClassColor(classId);
        // BGR -> RGB/RGBA
        return new SKColor((byte)scalar.Val2, (byte)scalar.Val1, (byte)scalar.Val0);
    }

    /// <summary>
    /// EN: Draws a label with background on the Mat.
    /// RU: Рисует метку с фоном на Mat.
    /// </summary>
    public static void DrawLabel(Mat mat, string text, Point topLeft, Scalar color)
    {
        int bl;
        var sz = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.45, 1, out bl);
        int ty = Math.Max(topLeft.Y - 4, sz.Height + 4);
        Cv2.Rectangle(mat,
            new Rect(topLeft.X, ty - sz.Height - 4, sz.Width + 6, sz.Height + 6),
            new Scalar(20, 20, 20), -1);
        Cv2.PutText(mat, text, new Point(topLeft.X + 3, ty),
            HersheyFonts.HersheySimplex, 0.45, color, 1, LineTypes.AntiAlias);
    }
}
