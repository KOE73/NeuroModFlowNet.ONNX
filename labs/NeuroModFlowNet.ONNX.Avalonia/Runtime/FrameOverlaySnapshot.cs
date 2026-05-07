namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Stores overlay data for one frame in source-frame coordinates.
/// RU: Хранит overlay-данные одного кадра в координатах исходного кадра.
/// </summary>
/// <remarks>
/// EN: The Skia overlay layer uses this snapshot to draw OBB boxes and text labels over the fitted video frame.
/// RU: Skia overlay layer использует этот snapshot, чтобы рисовать OBB-боксы и текстовые подписи поверх вписанного видео.
/// </remarks>
public sealed class FrameOverlaySnapshot
{
    public static FrameOverlaySnapshot Empty { get; } = new(0, 0, [], []);

    public FrameOverlaySnapshot(
        int frameWidth,
        int frameHeight,
        IReadOnlyList<OverlayObb> obbBoxes,
        IReadOnlyList<OverlayText> texts)
    {
        FrameWidth = frameWidth;
        FrameHeight = frameHeight;
        ObbBoxes = obbBoxes;
        Texts = texts;
    }

    public int FrameWidth { get; }
    public int FrameHeight { get; }
    public IReadOnlyList<OverlayObb> ObbBoxes { get; }
    public IReadOnlyList<OverlayText> Texts { get; }
}
