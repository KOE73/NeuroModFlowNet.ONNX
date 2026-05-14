using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Describes one oriented bounding box prepared for Skia overlay drawing.
/// RU: Описывает один oriented bounding box, подготовленный для Skia overlay-отрисовки.
/// </summary>
/// <remarks>
/// EN: Points and center are stored in source-frame coordinates. The painter maps them into the visible video viewport.
/// RU: Точки и центр хранятся в координатах исходного кадра. Painter переносит их в видимый video viewport.
/// </remarks>
public sealed class OverlayObb
{
    public OverlayObb(SKPoint center, SKPoint[] points, SKColor color, string label, bool fill = false)
    {
        Center = center;
        Points = points;
        Color = color;
        Label = label;
        Fill = fill;
    }

    public SKPoint Center { get; }
    public SKPoint[] Points { get; }
    public SKColor Color { get; }
    public string Label { get; }
    public bool Fill { get; }
}
