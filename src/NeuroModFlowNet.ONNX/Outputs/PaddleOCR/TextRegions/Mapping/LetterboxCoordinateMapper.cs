using System.Diagnostics;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Maps coordinates from a letterboxed model input image back to the source image.
/// RU: Возвращает координаты из letterbox-входа модели обратно в координаты исходного изображения.
/// </summary>
/// <remarks>
/// EN: This mapper intentionally keeps the common real-time path simple:
/// sourceX = (modelX - offsetX) / scale, sourceY = (modelY - offsetY) / scale.
/// More complex preprocessing chains can implement <see cref="IImageCoordinateMapper"/>
/// without changing text region extraction itself.
/// RU: Этот mapper специально оставляет основной realtime-путь простым:
/// sourceX = (modelX - offsetX) / scale, sourceY = (modelY - offsetY) / scale.
/// Более сложные preprocessing-цепочки могут реализовать <see cref="IImageCoordinateMapper"/>
/// без изменения самой вырезалки текстовых областей.
/// </remarks>
public readonly record struct LetterboxCoordinateMapper(float Scale, float OffsetX, float OffsetY) : IImageCoordinateMapper
{
    public static LetterboxCoordinateMapper Identity { get; } = new(1f, 0f, 0f);

    public static LetterboxCoordinateMapper Create(float scale, float offsetX, float offsetY)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(scale);
        return new LetterboxCoordinateMapper(scale, offsetX, offsetY);
    }

    public Point2f MapPointToSource(Point2f point)
    {
        Debug.Assert(Scale > 0);

        float inverseScale = 1f / Scale;
        return new Point2f(
            (point.X - OffsetX) * inverseScale,
            (point.Y - OffsetY) * inverseScale);
    }

    public void MapPointsToSource(ReadOnlySpan<Point2f> sourcePoints, Span<Point2f> destinationPoints)
    {
        Debug.Assert(Scale > 0);

        if(destinationPoints.Length < sourcePoints.Length)
            throw new ArgumentException("Destination must be at least as long as source.", nameof(destinationPoints));

        float inverseScale = 1f / Scale;

        for(int index = 0; index < sourcePoints.Length; index++)
        {
            // Letterbox preprocessing first scales the source image, then adds padding.
            // The inverse mapping therefore removes padding before applying inverse scale.
            Point2f point = sourcePoints[index];
            destinationPoints[index] = new Point2f(
                (point.X - OffsetX) * inverseScale,
                (point.Y - OffsetY) * inverseScale);
        }
    }
}
