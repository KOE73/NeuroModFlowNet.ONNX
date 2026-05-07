using SkiaSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Describes one text label prepared for Skia overlay drawing.
/// RU: Описывает одну текстовую подпись, подготовленную для Skia overlay-отрисовки.
/// </summary>
/// <remarks>
/// EN: The position is stored in source-frame coordinates and usually points to the text/ROI area detected by the OCR flow.
/// RU: Позиция хранится в координатах исходного кадра и обычно указывает на область текста или ROI, найденную OCR-потоком.
/// </remarks>
public sealed class OverlayText
{
    public OverlayText(SKPoint position, string text)
    {
        Position = position;
        Text = text;
    }

    public SKPoint Position { get; }
    public string Text { get; }
}
