using Avalonia;
using OpenCvSharp;
using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Provides helper methods for drawing BGRA OpenCV frames directly on a Skia canvas.
/// RU: Предоставляет helper-методы для прямой отрисовки BGRA OpenCV-кадров на Skia canvas.
/// </summary>
/// <remarks>
/// EN: The painter creates an <see cref="SKImage"/> over existing <see cref="Mat"/> memory and draws it fitted into the
/// Avalonia bounds. It expects a four-channel BGRA frame.
/// RU: Painter создает <see cref="SKImage"/> поверх существующей памяти <see cref="Mat"/> и вписывает его в границы
/// Avalonia. Ожидается четырехканальный BGRA-кадр.
/// </remarks>
internal static class SkiaMatPainter
{
    static readonly SKSamplingOptions Sampling = new(SKFilterMode.Linear, SKMipmapMode.None);

    public static void DrawMat(SKCanvas canvas, Mat mat, global::Avalonia.Rect bounds)
    {
        if(mat.Channels() != 4)
            return;

        var imageInfo = new SKImageInfo(mat.Width, mat.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
        using var pixmap = new SKPixmap(imageInfo, mat.Data, (int)mat.Step());
        using SKImage image = SKImage.FromPixels(pixmap);

        SKRect destination = FitRect(mat.Width, mat.Height, bounds);
        canvas.DrawImage(image, destination, Sampling);
    }

    public static SKRect FitRect(int sourceWidth, int sourceHeight, global::Avalonia.Rect bounds)
    {
        if(sourceWidth <= 0 || sourceHeight <= 0 || bounds.Width <= 0 || bounds.Height <= 0)
            return SKRect.Empty;

        double scale = Math.Min(bounds.Width / sourceWidth, bounds.Height / sourceHeight);
        float width = (float)(sourceWidth * scale);
        float height = (float)(sourceHeight * scale);
        float x = (float)(bounds.X + (bounds.Width - width) / 2);
        float y = (float)(bounds.Y + (bounds.Height - height) / 2);

        return new SKRect(x, y, x + width, y + height);
    }
}
