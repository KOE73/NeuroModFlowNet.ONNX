namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: A rectangle in image coordinates with an optional upright label. Color is packed as RRGGBBAA.
///
/// RU: Прямоугольник в координатах изображения с необязательной прямой подписью. Цвет упакован как RRGGBBAA.
/// </summary>
public readonly record struct Direct2DRectangle(
    float CenterX,
    float CenterY,
    float Width,
    float Height,
    float AngleDegrees,
    uint Rgba,
    string? Label = null,
    float StrokeWidth = 2);
