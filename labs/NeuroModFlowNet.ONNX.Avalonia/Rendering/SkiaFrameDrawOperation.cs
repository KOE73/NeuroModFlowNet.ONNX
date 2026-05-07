using Avalonia;
using Avalonia.Media;
using Avalonia.Platform;
using Avalonia.Rendering.SceneGraph;
using Avalonia.Skia;

namespace NeuroModFlowNet.ONNX.Avalonia.Rendering;

/// <summary>
/// EN: Represents the Avalonia custom draw operation that renders a <see cref="SkiaFrameView"/> on the Skia canvas.
/// RU: Представляет custom draw operation Avalonia, которая рисует <see cref="SkiaFrameView"/> на Skia canvas.
/// </summary>
/// <remarks>
/// EN: The operation obtains <c>ISkiaSharpApiLeaseFeature</c> from Avalonia and passes the leased canvas to the owning
/// frame control. It is intentionally small and contains no frame storage.
/// RU: Operation получает <c>ISkiaSharpApiLeaseFeature</c> из Avalonia и передает leased canvas владельцу-кадру. Она
/// намеренно маленькая и не хранит данные кадра.
/// </remarks>
internal sealed class SkiaFrameDrawOperation : ICustomDrawOperation
{
    readonly SkiaFrameView owner;

    public SkiaFrameDrawOperation(Rect bounds, SkiaFrameView owner)
    {
        Bounds = bounds;
        this.owner = owner;
    }

    public Rect Bounds { get; }

    public bool HitTest(Point point) => Bounds.Contains(point);

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
