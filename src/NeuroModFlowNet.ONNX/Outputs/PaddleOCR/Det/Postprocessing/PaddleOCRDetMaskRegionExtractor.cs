using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Converts PaddleOCR detection score maps into OCR quadrilateral regions.
/// RU: Преобразует score map PaddleOCR detection в четырехточечные OCR-регионы.
/// </summary>
/// <remarks>
/// EN: This class is the Det-specific branch before the common OCR path. It accepts the raw floating score map,
/// extracts local text components with contour geometry and returns <see cref="OcrQuadRegion"/> so downstream
/// crop, region filtering and recognition stay shared with YOLO OBB sources.
/// RU: Этот класс является Det-specific веткой до общего OCR-пути. Он принимает raw floating score map,
/// извлекает локальные текстовые компоненты через contour geometry и возвращает <see cref="OcrQuadRegion"/>,
/// чтобы дальнейшие crop, фильтрация регионов и recognition оставались общими с YOLO OBB источниками.
/// </remarks>
public sealed class PaddleOCRDetMaskRegionExtractor
{
    public static PaddleOCRDetMaskRegionExtractor Shared { get; } = new();

    #region Public API

    public int Extract(
        Mat scoreMap,
        in PaddleOCRDetMaskRegionExtractorOptions options,
        List<OcrQuadRegion> destinationRegions)
    {
        ArgumentNullException.ThrowIfNull(scoreMap);
        ArgumentNullException.ThrowIfNull(destinationRegions);
        ValidateInput(scoreMap, options);

        using Mat binaryMask = CreateBinaryMask(scoreMap, options.BitmapThreshold);
        Cv2.FindContours(
            binaryMask,
            out Point[][] contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        int writtenCount = 0;
        int candidateCount = Math.Min(contours.Length, options.MaxCandidateCount);
        for(int contourIndex = 0; contourIndex < candidateCount; contourIndex++)
        {
            Point[] contour = contours[contourIndex];
            if(contour.Length < 3) continue;
            if(!TryExtractRegion(scoreMap, contour, options, out OcrQuadRegion region)) continue;

            destinationRegions.Add(region);
            writtenCount++;
        }

        return writtenCount;
    }

    #endregion

    #region Extraction

    private static bool TryExtractRegion(
        Mat scoreMap,
        Point[] contour,
        in PaddleOCRDetMaskRegionExtractorOptions options,
        out OcrQuadRegion region)
    {
        Point2f[] contourPoints = ConvertContour(contour);
        RotatedRect rect = Cv2.MinAreaRect(contourPoints);
        Point2f[] points = rect.Points();
        if(GetShortSide(points) < options.MinimumBoxSide)
        {
            region = default;
            return false;
        }

        float score = ComputeBoxScore(scoreMap, points);
        if(score < options.BoxScoreThreshold)
        {
            region = default;
            return false;
        }

        if(options.EnableUnclip)
        {
            points = ExpandBox(points, options.UnclipRatio);
            if(points.Length < 4 || GetShortSide(points) < options.MinimumBoxSide)
            {
                region = default;
                return false;
            }
        }

        rect = Cv2.MinAreaRect(points);
        region = OcrQuadRegion.FromPoints(rect.Points());
        return true;
    }

    private static Point2f[] ConvertContour(Point[] contour)
    {
        var points = new Point2f[contour.Length];
        for(int index = 0; index < contour.Length; index++)
            points[index] = new Point2f(contour[index].X, contour[index].Y);

        return points;
    }

    private static float ComputeBoxScore(Mat scoreMap, ReadOnlySpan<Point2f> points)
    {
        Rect bounds = Cv2.BoundingRect(ToPointArray(points));
        bounds = ClipRect(bounds, scoreMap.Width, scoreMap.Height);
        if(bounds.Width <= 0 || bounds.Height <= 0) return 0f;

        using Mat localMask = Mat.Zeros(bounds.Height, bounds.Width, MatType.CV_8UC1);
        Point[] localPoints = new Point[points.Length];
        for(int index = 0; index < points.Length; index++)
            localPoints[index] = new Point(
                (int)MathF.Round(points[index].X) - bounds.X,
                (int)MathF.Round(points[index].Y) - bounds.Y);

        Cv2.FillPoly(localMask, [localPoints], Scalar.White);
        using Mat scoreRoi = scoreMap[bounds];
        return (float)Cv2.Mean(scoreRoi, localMask).Val0;
    }

    #endregion

    #region Unclip

    private static Point2f[] ExpandBox(ReadOnlySpan<Point2f> points, float unclipRatio)
    {
        Point2f center = GetCentroid(points);
        float area = MathF.Abs(GetPolygonArea(points));
        float perimeter = GetPerimeter(points);
        if(area <= float.Epsilon || perimeter <= float.Epsilon)
            return ToPoint2fArray(points);

        float distance = area * unclipRatio / perimeter;
        Point2f[] expandedPoints = new Point2f[points.Length];

        for(int index = 0; index < points.Length; index++)
        {
            Point2f direction = Normalize(new Point2f(points[index].X - center.X, points[index].Y - center.Y));
            expandedPoints[index] = new Point2f(
                points[index].X + direction.X * distance,
                points[index].Y + direction.Y * distance);
        }

        return expandedPoints;
    }

    #endregion

    #region Validation And Mat Helpers

    private static void ValidateInput(Mat scoreMap, in PaddleOCRDetMaskRegionExtractorOptions options)
    {
        if(scoreMap.Type() != MatType.CV_32FC1)
            throw new InvalidOperationException($"PaddleOCR Det region extraction expects {MatType.CV_32FC1}, but got {scoreMap.Type()}.");

        ArgumentOutOfRangeException.ThrowIfNegative(options.BitmapThreshold);
        ArgumentOutOfRangeException.ThrowIfNegative(options.BoxScoreThreshold);
        ArgumentOutOfRangeException.ThrowIfNegative(options.MinimumBoxSide);
        ArgumentOutOfRangeException.ThrowIfNegative(options.MaxCandidateCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.UnclipRatio);
    }

    private static Mat CreateBinaryMask(Mat scoreMap, float threshold)
    {
        var binaryMask = new Mat();
        Cv2.Threshold(scoreMap, binaryMask, threshold, 255.0, ThresholdTypes.Binary);
        binaryMask.ConvertTo(binaryMask, MatType.CV_8UC1);
        return binaryMask;
    }

    private static Rect ClipRect(Rect rect, int width, int height)
    {
        int x = Math.Clamp(rect.X, 0, width - 1);
        int y = Math.Clamp(rect.Y, 0, height - 1);
        int right = Math.Clamp(rect.Right, x + 1, width);
        int bottom = Math.Clamp(rect.Bottom, y + 1, height);
        return new Rect(x, y, right - x, bottom - y);
    }

    #endregion

    #region Geometry Helpers

    private static Point[] ToPointArray(ReadOnlySpan<Point2f> points)
    {
        Point[] result = new Point[points.Length];
        for(int index = 0; index < points.Length; index++)
            result[index] = new Point((int)MathF.Round(points[index].X), (int)MathF.Round(points[index].Y));

        return result;
    }

    private static Point2f[] ToPoint2fArray(ReadOnlySpan<Point2f> points)
    {
        Point2f[] result = new Point2f[points.Length];
        points.CopyTo(result);
        return result;
    }

    private static float GetShortSide(ReadOnlySpan<Point2f> points)
    {
        float first = Distance(points[0], points[1]);
        float second = Distance(points[1], points[2]);
        return MathF.Min(first, second);
    }

    private static Point2f GetCentroid(ReadOnlySpan<Point2f> points)
    {
        float x = 0f;
        float y = 0f;
        for(int index = 0; index < points.Length; index++)
        {
            x += points[index].X;
            y += points[index].Y;
        }

        return new Point2f(x / points.Length, y / points.Length);
    }

    private static Point2f Normalize(Point2f vector)
    {
        float length = MathF.Sqrt(vector.X * vector.X + vector.Y * vector.Y);
        return length <= float.Epsilon
            ? new Point2f(0f, 0f)
            : new Point2f(vector.X / length, vector.Y / length);
    }

    private static float GetPerimeter(ReadOnlySpan<Point2f> points)
    {
        float perimeter = 0f;
        for(int index = 0; index < points.Length; index++)
            perimeter += Distance(points[index], points[(index + 1) % points.Length]);

        return perimeter;
    }

    private static float Distance(Point2f first, Point2f second)
    {
        float x = first.X - second.X;
        float y = first.Y - second.Y;
        return MathF.Sqrt(x * x + y * y);
    }

    private static float GetPolygonArea(ReadOnlySpan<Point2f> points)
    {
        float area = 0f;
        for(int index = 0; index < points.Length; index++)
        {
            Point2f current = points[index];
            Point2f next = points[(index + 1) % points.Length];
            area += current.X * next.Y - next.X * current.Y;
        }

        return area * 0.5f;
    }

    #endregion
}
