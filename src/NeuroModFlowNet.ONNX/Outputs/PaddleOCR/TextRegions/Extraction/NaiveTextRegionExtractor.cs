using System.Diagnostics;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: CPU implementation for extracting OCR recognition crops from quadrilateral text regions.
/// RU: CPU-реализация вырезания OCR-областей распознавания из четырехугольных текстовых регионов.
/// </summary>
/// <remarks>
/// EN: This class is intentionally simple and predictable: order four points, run perspective warp,
/// normalize orientation, then resize with padding to the recognition model input size. It is the baseline
/// implementation for correctness and for realtime measurements before adding SIMD, tensor, pooled-buffer,
/// GPU, or ONNX-specialized variants.
/// RU: Класс специально оставлен простым и предсказуемым: упорядочить четыре точки, выполнить perspective warp,
/// нормализовать ориентацию и затем привести размер с padding под вход OCR-модели. Это базовая реализация
/// для проверки корректности и realtime-измерений перед SIMD, tensor, pooled-buffer, GPU или ONNX-вариантами.
/// </remarks>
public sealed class NaiveTextRegionExtractor : ITextRegionExtractor
{
    public static NaiveTextRegionExtractor Shared { get; } = new();

    public bool TryExtract(Mat sourceMat, in OcrQuadRegion sourceRegion, in TextRegionExtractionOptions options, out Mat? recognitionRoi)
    {
        ArgumentNullException.ThrowIfNull(sourceMat);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.TargetWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.TargetHeight);

        recognitionRoi = null;
        if(sourceMat.Empty()) return false;

        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        sourceRegion.CopyTo(sourcePoints);

        OrderTextLinePoints(sourcePoints);
        if(!AllPointsInside(sourcePoints, sourceMat.Width, sourceMat.Height)) return false;

        float unwarpedWidth = MathF.Max(Distance(sourcePoints[0], sourcePoints[1]), Distance(sourcePoints[3], sourcePoints[2]));
        float unwarpedHeight = MathF.Max(Distance(sourcePoints[0], sourcePoints[3]), Distance(sourcePoints[1], sourcePoints[2]));
        if(unwarpedWidth < 2 || unwarpedHeight < 2) return false;

        Point2f[] sourcePointArray =
        [
            sourcePoints[0],
            sourcePoints[1],
            sourcePoints[2],
            sourcePoints[3],
        ];

        Point2f[] targetPoints =
        [
            new(0, 0),
            new(unwarpedWidth - 1, 0),
            new(unwarpedWidth - 1, unwarpedHeight - 1),
            new(0, unwarpedHeight - 1),
        ];

        using Mat transform = Cv2.GetPerspectiveTransform(sourcePointArray, targetPoints);
        using var unwarped = new Mat();
        Cv2.WarpPerspective(
            sourceMat,
            unwarped,
            transform,
            new Size((int)Math.Round(unwarpedWidth), (int)Math.Round(unwarpedHeight)));

        if(unwarped.Height > unwarped.Width)
            Cv2.Rotate(unwarped, unwarped, RotateFlags.Rotate90Clockwise);

        Mat resizedRoi = ResizeRecognitionImage(unwarped, options.TargetWidth, options.TargetHeight);
        if(options.ProcessingStage is null)
        {
            recognitionRoi = resizedRoi;
            return true;
        }

        recognitionRoi = options.ProcessingStage.Process(resizedRoi);
        resizedRoi.Dispose();
        return true;
    }

    public int ExtractMany(
        Mat sourceMat,
        ReadOnlySpan<OcrQuadRegion> sourceRegions,
        in TextRegionExtractionOptions options,
        List<Mat> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);

        int extractedCount = 0;
        foreach(OcrQuadRegion sourceRegion in sourceRegions)
        {
            if(!TryExtract(sourceMat, sourceRegion, options, out Mat? recognitionRoi)) continue;
            if(recognitionRoi is null) continue;

            destination.Add(recognitionRoi);
            extractedCount++;
        }

        return extractedCount;
    }

    public static Mat ResizeRecognitionImage(Mat source, int targetWidth, int targetHeight)
    {
        Debug.Assert(source is not null);
        Debug.Assert(targetWidth > 0);
        Debug.Assert(targetHeight > 0);

        var result = new Mat(
            targetHeight,
            targetWidth,
            source.Type(),
            Scalar.Black);

        if(source.Empty()) return result;

        double heightScale = targetHeight / (double)source.Height;
        int scaledWidth = Math.Max(1, (int)Math.Round(source.Width * heightScale));
        int scaledHeight = targetHeight;

        if(scaledWidth > targetWidth)
        {
            double widthScale = targetWidth / (double)source.Width;
            scaledWidth = targetWidth;
            scaledHeight = Math.Max(1, (int)Math.Round(source.Height * widthScale));
        }

        using var resized = new Mat();
        Cv2.Resize(source, resized, new Size(scaledWidth, scaledHeight));

        int y = (targetHeight - scaledHeight) / 2;
        using Mat targetRoi = result[new Rect(0, y, scaledWidth, scaledHeight)];
        resized.CopyTo(targetRoi);

        return result;
    }

    private static void OrderTextLinePoints(Span<Point2f> points)
    {
        Point2f topLeft = points[0];
        Point2f bottomRight = points[0];
        Point2f topRight = points[0];
        Point2f bottomLeft = points[0];

        for(int index = 1; index < points.Length; index++)
        {
            Point2f point = points[index];
            if(point.X + point.Y < topLeft.X + topLeft.Y) topLeft = point;
            if(point.X + point.Y > bottomRight.X + bottomRight.Y) bottomRight = point;
            if(point.X - point.Y > topRight.X - topRight.Y) topRight = point;
            if(point.X - point.Y < bottomLeft.X - bottomLeft.Y) bottomLeft = point;
        }

        float topEdge = Distance(topLeft, topRight);
        float leftEdge = Distance(topLeft, bottomLeft);

        if(topEdge >= leftEdge)
        {
            points[0] = topLeft;
            points[1] = topRight;
            points[2] = bottomRight;
            points[3] = bottomLeft;
            return;
        }

        points[0] = bottomLeft;
        points[1] = topLeft;
        points[2] = topRight;
        points[3] = bottomRight;
    }

    private static bool AllPointsInside(ReadOnlySpan<Point2f> points, int width, int height)
    {
        foreach(Point2f point in points)
        {
            if(point.X < 0 || point.Y < 0 || point.X >= width || point.Y >= height)
                return false;
        }

        return true;
    }

    private static float Distance(Point2f first, Point2f second)
    {
        float x = first.X - second.X;
        float y = first.Y - second.Y;
        return MathF.Sqrt(x * x + y * y);
    }
}
