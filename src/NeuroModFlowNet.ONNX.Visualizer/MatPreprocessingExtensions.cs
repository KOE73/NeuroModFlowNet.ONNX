using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Visualizer;

public static class MatPreprocessingExtensions
{
    private static readonly Scalar DefaultPadColor = new Scalar(114, 114, 114);

    /// <summary>
    /// EN: Resizes the image to the target size while maintaining aspect ratio (Letterbox).
    /// <br/>
    /// RU: Изменяет размер изображения под целевой размер с сохранением пропорций и добавлением полей (Letterbox).
    /// </summary>
    /// <param name="src">Source image.</param>
    /// <param name="targetWidth">Target width.</param>
    /// <param name="targetHeight">Target height.</param>
    /// <param name="info">Scaling and padding information for mapping results back.</param>
    /// <param name="padColor">Padding color (default is gray 114, 114, 114).</param>
    /// <returns>New Mat with target size.</returns>
    public static Mat Letterbox(this Mat src, int targetWidth, int targetHeight, out LetterboxInfo info, Scalar? padColor = null)
    {
        float r = Math.Min((float)targetWidth / src.Width, (float)targetHeight / src.Height);
        int newUnpadW = (int)Math.Round(src.Width * r);
        int newUnpadH = (int)Math.Round(src.Height * r);

        int offsetX = (targetWidth - newUnpadW) / 2;
        int offsetY = (targetHeight - newUnpadH) / 2;

        info = new LetterboxInfo
        {
            Ratio = r,
            OffsetX = offsetX,
            OffsetY = offsetY,
            SourceWidth = src.Width,
            SourceHeight = src.Height,
            TargetWidth = targetWidth,
            TargetHeight = targetHeight
        };

        Mat resized = new Mat();
        Cv2.Resize(src, resized, new Size(newUnpadW, newUnpadH));

        Mat dst = new Mat(targetHeight, targetWidth, src.Type(), padColor ?? DefaultPadColor);
        resized.CopyTo(dst[new Rect(offsetX, offsetY, newUnpadW, newUnpadH)]);
        resized.Dispose();

        return dst;
    }
}

public struct LetterboxInfo
{
    public float Ratio;
    public int OffsetX;
    public int OffsetY;
    public int SourceWidth;
    public int SourceHeight;
    public int TargetWidth;
    public int TargetHeight;

    /// <summary>
    /// EN: Maps coordinates from the letterboxed image back to the source image.
    /// <br/>
    /// RU: Пересчитывает координаты из letterboxed изображения обратно в исходное.
    /// </summary>
    public Point2f MapBack(float x, float y)
    {
        return new Point2f((x - OffsetX) / Ratio, (y - OffsetY) / Ratio);
    }
    
    /// <summary>
    /// EN: Maps scale factor for width/height from letterboxed image back to source image.
    /// <br/>
    /// RU: Пересчитывает коэффициент масштабирования длины/ширины из letterboxed изображения обратно в исходное.
    /// </summary>
    public float MapScale(float value) => value / Ratio;
}
