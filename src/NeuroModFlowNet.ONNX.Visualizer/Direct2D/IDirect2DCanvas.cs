namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Draws on the current GPU target in input-image pixels. Colors are packed as RRGGBBAA.
///
/// RU: Рисует на текущей GPU-поверхности в пикселях входного изображения. Цвета упакованы как RRGGBBAA.
/// </summary>
/// <remarks>
/// EN: Use the canvas only during the active drawing pass on the thread that opened it.
///
/// RU: Используйте холст только во время активного прохода в потоке, который его открыл.
/// </remarks>
public interface IDirect2DCanvas
{
    int Width { get; }
    int Height { get; }

    void DrawRectangle(float centerX, float centerY, float width, float height, float angleDegrees, uint rgba,
        float strokeWidth = 2);

    void DrawLine(float startX, float startY, float endX, float endY, uint rgba, float strokeWidth = 2);

    /// <summary>
    /// EN: Draws clipped text in the specified layout bounds. A font size of zero selects the default size.
    ///
    /// RU: Рисует текст с обрезкой по границам указанной области. Нулевой размер шрифта выбирает размер по умолчанию.
    /// </summary>
    void DrawText(string text, float left, float top, float width, float height, uint rgba, float fontSize = 0);
}
