using Avalonia;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;
using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Provides Skia drawing code for OBB boxes, centers, and OCR text labels.
/// RU: Предоставляет Skia-отрисовку OBB-боксов, центров и OCR-текстовых подписей.
/// </summary>
/// <remarks>
/// EN: The painter receives already prepared overlay data in source-frame coordinates and maps it into the fitted video
/// viewport. It contains drawing logic only and does not call neural network code.
/// RU: Painter получает уже подготовленные overlay-данные в координатах исходного кадра и переносит их во вписанный
/// video viewport. Внутри только отрисовка, без вызовов нейросети.
/// </remarks>
internal static class SkiaOverlayPainter
{
    public static void Draw(SKCanvas canvas, Rect bounds, FrameOverlaySnapshot snapshot)
    {
        if(snapshot.FrameWidth <= 0 || snapshot.FrameHeight <= 0) return;

        SKRect frameRect = SkiaMatPainter.FitRect(snapshot.FrameWidth, snapshot.FrameHeight, bounds);
        if(frameRect.IsEmpty) return;

        float scale = frameRect.Width / snapshot.FrameWidth;
        using var strokePaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = Math.Max(1.5f, 2f * scale),
            IsAntialias = true,
        };

        using var fillPaint = new SKPaint
        {
            Style = SKPaintStyle.Fill,
            IsAntialias = true,
        };

        using var centerPaint = new SKPaint
        {
            Style = SKPaintStyle.Fill,
            Color = SKColors.Red,
            IsAntialias = true,
        };

        using var textPaint = new SKPaint
        {
            Color = new SKColor(255, 255, 255, 220),
            IsAntialias = true,
        };

        using var textOutlinePaint = new SKPaint
        {
            Color = new SKColor(0, 0, 0, 150),
            IsAntialias = true,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 2,
        };

        using var backgroundPaint = new SKPaint
        {
            Color = new SKColor(10, 10, 10, 130),
            Style = SKPaintStyle.Fill,
            IsAntialias = true,
        };

        using var typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold);
        using var font = new SKFont(typeface, Math.Max(9, 11 * scale));

        foreach(OverlayObb box in snapshot.ObbBoxes)
        {
            strokePaint.Color = box.Color;
            using var path = new SKPath();

            for(int index = 0; index < box.Points.Length; index++)
            {
                SKPoint point = MapPoint(box.Points[index], frameRect, scale);
                if(index == 0)
                    path.MoveTo(point);
                else
                    path.LineTo(point);
            }

            path.Close();
            if(box.Fill)
            {
                fillPaint.Color = box.Color;
                canvas.DrawPath(path, fillPaint);
            }
            else
            {
                strokePaint.Color = box.Color;
                canvas.DrawPath(path, strokePaint);
                canvas.DrawCircle(MapPoint(box.Center, frameRect, scale), Math.Max(2, 3 * scale), centerPaint);
                DrawLabel(canvas, box.Label, MapPoint(box.Points[0], frameRect, scale), font, textPaint, textOutlinePaint, backgroundPaint);
            }
        }

        foreach(OverlayText text in snapshot.Texts)
            DrawLabel(canvas, text.Text, MapPoint(text.Position, frameRect, scale), font, textPaint, textOutlinePaint, backgroundPaint);
    }

    private static SKPoint MapPoint(SKPoint point, SKRect frameRect, float scale) =>
        new(frameRect.Left + point.X * scale, frameRect.Top + point.Y * scale);

    private static void DrawLabel(
        SKCanvas canvas,
        string text,
        SKPoint topLeft,
        SKFont font,
        SKPaint textPaint,
        SKPaint textOutlinePaint,
        SKPaint backgroundPaint)
    {
        if(string.IsNullOrWhiteSpace(text)) return;

        font.MeasureText(text, out SKRect textBounds, textPaint);
        float padding = 1f;
        float y = Math.Max(topLeft.Y - 2, textBounds.Height + padding);
        var backgroundRect = new SKRect(
            topLeft.X,
            y - textBounds.Height - padding,
            topLeft.X + textBounds.Width + padding * 2,
            y + padding);

        canvas.DrawRoundRect(backgroundRect, 2, 2, backgroundPaint);
        canvas.DrawText(text, topLeft.X + padding, y, font, textOutlinePaint);
        canvas.DrawText(text, topLeft.X + padding, y, font, textPaint);
    }
}
