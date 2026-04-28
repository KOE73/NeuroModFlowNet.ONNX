using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Utility class for visualizing box detection results using OpenCV and SkiaSharp.
/// <br/>
/// RU: Утилитарный класс для визуализации результатов детекции боксов с помощью OpenCV и SkiaSharp.
/// </summary>
public static class BoxPainter
{
    public static Scalar ClassColor(int classId) => VisualUtils.ClassColor(classId);
    public static SKColor ClassColorSkia(int classId) => VisualUtils.ClassColorSkia(classId);

    /// <summary>
    /// EN: Draws a label with background on the Mat.
    /// RU: Рисует метку с фоном на Mat.
    /// </summary>
    public static void DrawLabel(Mat mat, string text, Point topLeft, Scalar color) 
        => VisualUtils.DrawLabel(mat, text, topLeft, color);

    /// <summary>
    /// EN: Draws YOLO Box detections on Mat using OpenCV.
    /// RU: Рисует YOLO Box детекции на Mat с помощью OpenCV.
    /// </summary>
    public static void DrawBox<T>(
        Mat mat,
        T[] boxes,
        LetterboxInfo info,
        float scaleX = 1.0f, float scaleY = 1.0f,
        Func<int, string>? nameResolver = null,
        bool drawLabel = true) where T : struct, IOutAsT<YoloBox>
    {
        foreach(var item in boxes)
        {
            var box = item.AsStd();
            var p1 = info.MapBack(box.X, box.Y);
            var p2 = info.MapBack(box.W, box.H);

            int x1 = (int)(p1.X * scaleX);
            int y1 = (int)(p1.Y * scaleY);
            int x2 = (int)(p2.X * scaleX);
            int y2 = (int)(p2.Y * scaleY);

            var color = ClassColor((int)box.Class);
            var r = new Rect(x1, y1, x2 - x1, y2 - y1);
            Cv2.Rectangle(mat, r, color, 2, LineTypes.AntiAlias);

            if(drawLabel)
            {
                string name = nameResolver?.Invoke((int)box.Class) ?? $"#{(int)box.Class}";
                string lbl = $"{name} {box.Score:P0}";
                DrawLabel(mat, lbl, r.TopLeft, color);
            }
        }
    }

    /// <summary>
    /// EN: SkiaSharp implementation of DrawBox - zero-copy, directly in Mat memory.
    /// <br/>
    /// RU: Реализация DrawBox на SkiaSharp - без копирования, напрямую в памяти Mat.
    /// </summary>
    public static void DrawBoxSkia<T>(
        Mat mat,
        T[] boxes,
        LetterboxInfo info,
        float scaleX = 1.0f, float scaleY = 1.0f,
        Func<int, string>? nameResolver = null,
        bool drawLabel = true) where T : struct, IOutAsT<YoloBox>
    {
        // OpenCV Mat (CV_8UC3) is BGR. Skia needs BGRA8888 for zero-copy if Mat is BGRA.
        // If Mat is BGR (8UC3), zero-copy is tricky because Skia expects RGBA/BGRA (4 bytes).
        // However, if the Mat data is passed, we must ensure SKColorType matches.
        
        SKColorType colorType = SKColorType.Bgra8888;
        if (mat.Channels() == 3)
        {
            // Note: Skia doesn't have a direct BGR888 (3 bytes) equivalent that works easily on all platforms.
            // In the original code, it was assumed mat is BGRA (after CvtColor).
            colorType = SKColorType.Bgra8888;
        }

        var infoSk = new SKImageInfo(mat.Width, mat.Height, colorType, SKAlphaType.Premul);

        using var surface = SKSurface.Create(infoSk, mat.Data, (int)mat.Step());
        if(surface == null) return;

        var canvas = surface.Canvas;

        using var paintRect = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 2,
            IsAntialias = true
        };

        using var typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold);
        using var font = new SKFont(typeface, 14);

        using var paintText = new SKPaint
        {
            Color = SKColors.White,
            IsAntialias = true,
        };

        using var paintBg = new SKPaint
        {
            Color = new SKColor(20, 20, 20, 200),
            Style = SKPaintStyle.Fill
        };

        foreach(var item in boxes)
        {
            var box = item.AsStd();
            var p1 = info.MapBack(box.X, box.Y);
            var p2 = info.MapBack(box.W, box.H);

            float x1 = p1.X * scaleX;
            float y1 = p1.Y * scaleY;
            float x2 = p2.X * scaleX;
            float y2 = p2.Y * scaleY;

            var color = ClassColorSkia((int)box.Class);
            paintRect.Color = color;

            canvas.DrawRect(x1, y1, x2 - x1, y2 - y1, paintRect);

            if(drawLabel)
            {
                string name = nameResolver?.Invoke((int)box.Class) ?? $"#{(int)box.Class}";
                string lbl = $"{name} {box.Score:P0}";

                var textRect = new SKRect();
                font.MeasureText(lbl, out textRect, paintText);

                float ty = Math.Max(y1 - 4, textRect.Height + 4);
                var bgRect = new SKRect(x1, ty - textRect.Height - 4, x1 + textRect.Width + 6, ty + 2);

                canvas.DrawRect(bgRect, paintBg);
                canvas.DrawText(lbl, x1 + 3, ty, font, paintText);
            }
        }
        canvas.Flush();
    }
}
