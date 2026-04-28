namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Utility class for visualizing Oriented Bounding Box (OBB) detection results.
/// RU: Утилитарный класс для визуализации результатов детекции поворотных боксов (OBB).
/// </summary>
public static class ObbPainter
{
    public static void DrawObb(
        Mat mat,
        YoloObb[] boxes,
        LetterboxInfo info,
        float scaleX, float scaleY,
        Scalar? color = null,
        Func<int, string>? nameResolver = null)
    {
        foreach (var box in boxes)
        {
            var originalPos = info.MapBack(box.X, box.Y);
            float originalW = info.MapScale(box.W);
            float originalH = info.MapScale(box.H);

            float viewX = originalPos.X * scaleX;
            float viewY = originalPos.Y * scaleY;
            float viewW = originalW * scaleX;
            float viewH = originalH * scaleY;

            var rotatedRect = new RotatedRect(
                new Point2f(viewX, viewY),
                new Size2f(viewW, viewH),
                (float)(box.Angle * 180.0 / Math.PI));

            var vertices = rotatedRect.Points().Select(p => p.ToPoint()).ToArray();
            var drawColor = color ?? VisualUtils.ClassColor(box.Class);
            
            Cv2.Polylines(mat, [vertices], isClosed: true, color: drawColor, thickness: 2, lineType: LineTypes.AntiAlias);
            Cv2.Circle(mat, new Point((int)viewX, (int)viewY), 3, Scalar.Red, -1);
            
            string name = nameResolver?.Invoke(box.Class) ?? $"#{box.Class}";
            VisualUtils.DrawLabel(mat, $"{name} {box.Score:P0}", vertices[0], drawColor);
        }
    }

    public static void DrawObb(
        Mat mat,
        YoloObb_FP32_XYWHSCA[] boxes,
        LetterboxInfo info,
        float scaleX, float scaleY,
        Scalar? color = null,
        Func<int, string>? nameResolver = null)
    {
        foreach (var box in boxes)
        {
            var originalPos = info.MapBack(box.X, box.Y);
            float originalW = info.MapScale(box.W);
            float originalH = info.MapScale(box.H);

            float viewX = originalPos.X * scaleX;
            float viewY = originalPos.Y * scaleY;
            float viewW = originalW * scaleX;
            float viewH = originalH * scaleY;

            var rotatedRect = new RotatedRect(
                new Point2f(viewX, viewY),
                new Size2f(viewW, viewH),
                (float)(box.Angle * 180.0 / Math.PI));

            var vertices = rotatedRect.Points().Select(p => p.ToPoint()).ToArray();
            var drawColor = color ?? VisualUtils.ClassColor((int)box.Class);
            
            Cv2.Polylines(mat, [vertices], isClosed: true, color: drawColor, thickness: 2, lineType: LineTypes.AntiAlias);
            Cv2.Circle(mat, new Point((int)viewX, (int)viewY), 3, Scalar.Red, -1);
            
            string name = nameResolver?.Invoke((int)box.Class) ?? $"#{(int)box.Class}";
            VisualUtils.DrawLabel(mat, $"{name} {box.Score:P0}", vertices[0], drawColor);
        }
    }

    /// <summary>
    /// EN: Legacy overload (no letterbox info, direct coords).
    /// RU: Устаревший метод (без LetterboxInfo, прямые координаты).
    /// </summary>
    public static void DrawObb(Mat mat, YoloObb_FP32_XYWHSCA[] boxes, Scalar color, float scaleX = 1f, float scaleY = 1f)
    {
        foreach (var box in boxes)
        {
            if (box.Score < 0.3f) continue;
            var rotatedRect = new RotatedRect(
                new Point2f(box.X * scaleX, box.Y * scaleY),
                new Size2f(box.W * scaleX, box.H * scaleY),
                (float)(box.Angle * 180.0 / Math.PI));
            var vertices = rotatedRect.Points().Select(p => p.ToPoint()).ToArray();
            Cv2.Polylines(mat, [vertices], isClosed: true, color: color, thickness: 2, lineType: LineTypes.AntiAlias);
            Cv2.Circle(mat, new Point((int)(box.X * scaleX), (int)(box.Y * scaleY)), 2, Scalar.Red, -1);
            Cv2.PutText(mat, $"{box.Score:P0}", vertices[0],
                HersheyFonts.HersheySimplex, 0.4, color, 1, LineTypes.AntiAlias);
        }
    }
}
