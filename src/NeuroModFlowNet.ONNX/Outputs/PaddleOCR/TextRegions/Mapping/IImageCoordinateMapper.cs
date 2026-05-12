using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Maps points from the current processing coordinate space back to the original source image.
/// RU: Преобразует точки из текущей системы координат обработки обратно в исходное изображение.
/// </summary>
/// <remarks>
/// EN: The contract is point-based on purpose. Width, height, angle, and rectangles are derived from points
/// and can become ambiguous after scale, offset, rotation, crop, or chained preprocessing transforms.
/// Implementations may be cheap structs for realtime paths or full transform chains for more complex pipelines.
/// RU: Контракт намеренно основан только на точках. Ширина, высота, угол и прямоугольники вычисляются из точек
/// и становятся неоднозначными после scale, offset, rotation, crop или цепочки preprocessing-трансформов.
/// Реализации могут быть дешевыми struct для realtime-пути или полноценными цепочками трансформов.
/// </remarks>
public interface IImageCoordinateMapper
{
    Point2f MapPointToSource(Point2f point);

    void MapPointsToSource(ReadOnlySpan<Point2f> sourcePoints, Span<Point2f> destinationPoints)
    {
        if(destinationPoints.Length < sourcePoints.Length)
            throw new ArgumentException("Destination must be at least as long as source.", nameof(destinationPoints));

        for(int index = 0; index < sourcePoints.Length; index++)
            destinationPoints[index] = MapPointToSource(sourcePoints[index]);
    }
}
