using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Converts YOLO OBB detector output into source-image text regions for OCR extraction.
/// RU: Преобразует результат YOLO OBB-детектора в текстовые области исходного изображения для OCR-вырезания.
/// </summary>
/// <remarks>
/// EN: This is the intentionally narrow bridge between the application pipeline and the reusable crop library.
/// The cropper itself does not know about YOLO, but the realtime YOLO OBB -> crop -> OCR path still avoids
/// heavyweight adapter objects and converts directly into the four-point representation used downstream.
/// RU: Это намеренно узкий мост между прикладным pipeline и переиспользуемой библиотечной вырезалкой.
/// Сама вырезалка не знает про YOLO, но realtime-путь YOLO OBB -> crop -> OCR не тянет тяжелые adapter-объекты
/// и напрямую переводит данные в четырехточечное представление для следующих этапов.
/// </remarks>
public static class YoloObbOcrRegionMapper
{
    public static OcrQuadRegion MapToSourceRegion(
        in YoloObb box,
        in LetterboxCoordinateMapper mapper,
        float heightScale = 1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heightScale);

        Span<Point2f> modelPoints = stackalloc Point2f[4];
        Span<Point2f> sourcePoints = stackalloc Point2f[4];

        // YoloObb is kept as the fast application-level input, but the extractor itself
        // receives four source-image points. That keeps crop logic independent from the detector.
        var modelRegion = new OcrObbRegion(
            box.X,
            box.Y,
            Math.Max(2, box.W),
            Math.Max(2, box.H * heightScale),
            box.Angle);

        modelRegion.GetPoints(modelPoints);

        // Coordinate transforms operate on points only. Width, height, and angle are not
        // mapped separately because they are derived data and can be invalid for crop/rotate chains.
        mapper.MapPointsToSource(modelPoints, sourcePoints);
        return OcrQuadRegion.FromPoints(sourcePoints);
    }

    public static int MapToSourceRegions(
        ReadOnlySpan<YoloObb> boxes,
        in LetterboxCoordinateMapper mapper,
        float heightScale,
        Span<OcrQuadRegion> destinationRegions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heightScale);

        if(destinationRegions.Length < boxes.Length)
            throw new ArgumentException("Destination must be at least as long as source boxes.", nameof(destinationRegions));

        for(int index = 0; index < boxes.Length; index++)
            destinationRegions[index] = MapToSourceRegion(boxes[index], mapper, heightScale);

        return boxes.Length;
    }
}
