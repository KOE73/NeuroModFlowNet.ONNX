using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Represents an Avalonia control that displays one OpenCV <see cref="Mat"/> frame through Skia drawing.
/// RU: Представляет Avalonia control, который отображает один OpenCV <see cref="Mat"/> кадр через Skia-отрисовку.
/// </summary>
/// <remarks>
/// EN: The control accepts BGRA, BGR, and grayscale frames. Internally it stores a BGRA copy and renders it without
/// encoding the frame to PNG, JPG, stream, or Avalonia Bitmap. Call <see cref="UpdateFrame"/> from the UI thread.
/// RU: Control принимает BGRA, BGR и grayscale кадры. Внутри хранит BGRA-копию и рисует ее без кодирования кадра в
/// PNG, JPG, stream или Avalonia Bitmap. Вызывайте <see cref="UpdateFrame"/> из UI thread.
/// </remarks>
public sealed class SkiaFrameView : Control, IDisposable
{
    readonly object syncRoot = new();
    Mat? frame;

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        context.Custom(new SkiaFrameDrawOperation(new global::Avalonia.Rect(Bounds.Size), this));
    }

    public void UpdateFrame(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        lock(syncRoot)
        {
            frame ??= new Mat();
            if(source.Channels() == 4)
                source.CopyTo(frame);
            else if(source.Channels() == 3)
                Cv2.CvtColor(source, frame, ColorConversionCodes.BGR2BGRA);
            else if(source.Channels() == 1)
                Cv2.CvtColor(source, frame, ColorConversionCodes.GRAY2BGRA);
            else
                source.CopyTo(frame);
        }

        InvalidateVisual();
    }

    internal void Draw(SkiaSharp.SKCanvas canvas, global::Avalonia.Rect bounds)
    {
        lock(syncRoot)
        {
            if(frame is null || frame.Empty()) return;
            SkiaMatPainter.DrawMat(canvas, frame, bounds);
        }
    }

    public void Dispose()
    {
        lock(syncRoot)
        {
            frame?.Dispose();
            frame = null;
        }
    }
}
