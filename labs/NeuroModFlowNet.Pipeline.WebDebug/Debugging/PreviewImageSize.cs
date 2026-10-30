using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal static class PreviewImageSize
{
    public const int MaximumEdge = 8192;
    public const long MaximumPixels = 16 * 1024 * 1024;
    public const int MaximumEncodedBytes = 64 * 1024 * 1024;

    public static bool IsValid(int maxEdge) => maxEdge == 0 || maxEdge is >= 100 and <= MaximumEdge;

    public static Size Resolve(int width, int height, int maxEdge)
    {
        if(!IsValid(maxEdge)) throw new ArgumentOutOfRangeException(nameof(maxEdge), "Use 0 for original size or 100–8192 pixels.");
        if(width <= 0 || height <= 0) throw new ArgumentException("Image dimensions must be positive.");
        double scale = maxEdge == 0 ? 1 : Math.Min(1, maxEdge / (double)Math.Max(width, height));
        var size = new Size(Math.Max(1, (int)(width * scale)), Math.Max(1, (int)(height * scale)));
        if((long)size.Width * size.Height > MaximumPixels) throw new InvalidOperationException("Preview exceeds the 16 megapixel limit. Choose a smaller image size.");
        return size;
    }

    public static Mat CopyCpu(Mat image, int maxEdge)
    {
        Size size = Resolve(image.Width, image.Height, maxEdge);
        if(size == image.Size()) return image.Clone();
        var result = new Mat();
        try { Cv2.Resize(image, result, size); return result; }
        catch { result.Dispose(); throw; }
    }
}
