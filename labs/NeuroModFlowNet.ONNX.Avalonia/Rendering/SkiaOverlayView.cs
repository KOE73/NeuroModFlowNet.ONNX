using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Represents an Avalonia control that draws neural network overlays above the live video frame.
/// RU: Представляет Avalonia control, который рисует overlay нейросети поверх live-видео.
/// </summary>
/// <remarks>
/// EN: The control stores the latest overlay snapshot and draws OBB boxes and text labels through a separate Skia custom
/// draw operation. It does not own the video frame itself.
/// RU: Control хранит последний snapshot overlay и рисует OBB-боксы и текстовые подписи через отдельную Skia custom
/// draw operation. Самим видео-кадром он не владеет.
/// </remarks>
public sealed class SkiaOverlayView : Control
{
    readonly object syncRoot = new();
    FrameOverlaySnapshot overlay = FrameOverlaySnapshot.Empty;

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        context.Custom(new SkiaOverlayDrawOperation(new Rect(Bounds.Size), this));
    }

    public void UpdateOverlay(FrameOverlaySnapshot snapshot)
    {
        lock(syncRoot)
            overlay = snapshot;

        InvalidateVisual();
    }

    internal void Draw(SkiaSharp.SKCanvas canvas, Rect bounds)
    {
        FrameOverlaySnapshot snapshot;
        lock(syncRoot)
            snapshot = overlay;

        SkiaOverlayPainter.Draw(canvas, bounds, snapshot);
    }
}
