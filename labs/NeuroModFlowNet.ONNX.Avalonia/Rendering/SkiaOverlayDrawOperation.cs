using Avalonia;
using Avalonia.Media;
using Avalonia.Platform;
using Avalonia.Rendering.SceneGraph;
using Avalonia.Skia;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Represents the Avalonia custom draw operation that renders the overlay layer through Skia.
/// RU: Представляет custom draw operation Avalonia, которая рисует overlay-слой через Skia.
/// </summary>
/// <remarks>
/// EN: The operation leases Avalonia's Skia canvas and delegates all overlay painting to the owning
/// <see cref="SkiaOverlayView"/>. Hit testing is disabled because the current overlay is visual-only.
/// RU: Operation берет Skia canvas у Avalonia и передает всю отрисовку overlay владельцу
/// <see cref="SkiaOverlayView"/>. Hit testing выключен, потому что текущий overlay только визуальный.
/// </remarks>
internal sealed class SkiaOverlayDrawOperation : ICustomDrawOperation
{
    readonly SkiaOverlayView owner;

    public SkiaOverlayDrawOperation(Rect bounds, SkiaOverlayView owner)
    {
        Bounds = bounds;
        this.owner = owner;
    }

    public Rect Bounds { get; }

    public bool HitTest(Point point) => false;

    public void Render(ImmediateDrawingContext context)
    {
        ISkiaSharpApiLeaseFeature? leaseFeature = context.TryGetFeature<ISkiaSharpApiLeaseFeature>();
        if(leaseFeature is null) return;

        using ISkiaSharpApiLease lease = leaseFeature.Lease();
        owner.Draw(lease.SkCanvas, Bounds);
    }

    public void Dispose()
    {
    }

    public bool Equals(ICustomDrawOperation? other) => false;
}
