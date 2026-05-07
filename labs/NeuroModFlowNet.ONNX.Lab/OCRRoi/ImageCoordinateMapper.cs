using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public static class ImageCoordinateMapper
{
    public static List<RotatedRect> MapYoloObbToSourceRects(
        ReadOnlySpan<YoloObb> boxes,
        ImageResizeTransform transform,
        float heightScale = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(transform.Scale);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heightScale);

        List<RotatedRect> sourceRects = new(boxes.Length);

        foreach(YoloObb box in boxes)
            sourceRects.Add(MapYoloObbToSourceRect(box, transform, heightScale));

        return sourceRects;
    }

    public static RotatedRect MapYoloObbToSourceRect(
        YoloObb box,
        ImageResizeTransform transform,
        float heightScale = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(transform.Scale);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heightScale);

        Point2f sourcePosition = transform.MapPointToSource(box.X, box.Y);
        float sourceWidth = Math.Max(2, transform.MapLengthToSource(box.W));
        float sourceHeight = Math.Max(2, transform.MapLengthToSource(box.H) * heightScale);

        return new RotatedRect(
            sourcePosition,
            new Size2f(sourceWidth, sourceHeight),
            (float)(box.Angle * 180.0 / Math.PI));
    }
}
