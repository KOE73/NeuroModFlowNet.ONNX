using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public static class OcrRoiExtractor
{
    public static List<Mat> PrepareRecognitionImages(
        Mat sourceMat,
        ReadOnlySpan<RotatedRect> sourceRects,
        int targetWidth,
        int targetHeight,
        IOcrRoiProcessingStage? processingStage = null)
    {
        List<Mat> preparedImages = [];

        foreach(RotatedRect sourceRect in sourceRects)
        {
            Mat? recognitionRoi = TryExtractRecognitionRoi(
                sourceMat,
                sourceRect,
                targetWidth,
                targetHeight,
                processingStage);

            if(recognitionRoi is null) continue;

            preparedImages.Add(recognitionRoi);
        }

        return preparedImages;
    }

    public static Mat? TryExtractRecognitionRoi(
        Mat sourceMat,
        RotatedRect rotatedRect,
        int targetWidth,
        int targetHeight,
        IOcrRoiProcessingStage? processingStage = null)
    {
        ArgumentNullException.ThrowIfNull(sourceMat);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetHeight);

        if(sourceMat.Empty()) return null;

        Point2f[] sourcePoints = OrderTextLinePoints(rotatedRect.Points());
        if(!AllPointsInside(sourcePoints, sourceMat.Width, sourceMat.Height)) return null;

        float unwarpedWidth = Math.Max(rotatedRect.Size.Width, rotatedRect.Size.Height);
        float unwarpedHeight = Math.Min(rotatedRect.Size.Width, rotatedRect.Size.Height);
        if(unwarpedWidth < 2 || unwarpedHeight < 2) return null;

        Point2f[] targetPoints =
        [
            new(0, 0),
            new(unwarpedWidth - 1, 0),
            new(unwarpedWidth - 1, unwarpedHeight - 1),
            new(0, unwarpedHeight - 1),
        ];

        using Mat transform = Cv2.GetPerspectiveTransform(sourcePoints, targetPoints);
        using var unwarped = new Mat();
        Cv2.WarpPerspective(
            sourceMat,
            unwarped,
            transform,
            new Size((int)Math.Round(unwarpedWidth), (int)Math.Round(unwarpedHeight)));

        if(unwarped.Height > unwarped.Width)
            Cv2.Rotate(unwarped, unwarped, RotateFlags.Rotate90Clockwise);

        Mat recognitionRoi = ResizeRecognitionRoi(unwarped, targetWidth, targetHeight);
        if(processingStage is null) return recognitionRoi;

        Mat processedRecognitionRoi = processingStage.Process(recognitionRoi);
        recognitionRoi.Dispose();
        return processedRecognitionRoi;
    }

    public static Mat ResizeRecognitionRoi(Mat source, int targetWidth, int targetHeight)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetHeight);

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

    private static Point2f[] OrderTextLinePoints(Point2f[] points)
    {
        Point2f topLeft = points.MinBy(point => point.X + point.Y);
        Point2f bottomRight = points.MaxBy(point => point.X + point.Y);
        Point2f topRight = points.MaxBy(point => point.X - point.Y);
        Point2f bottomLeft = points.MinBy(point => point.X - point.Y);

        float topEdge = Distance(topLeft, topRight);
        float leftEdge = Distance(topLeft, bottomLeft);

        if(topEdge >= leftEdge)
        {
            return
            [
                topLeft,
                topRight,
                bottomRight,
                bottomLeft,
            ];
        }

        return
        [
            bottomLeft,
            topLeft,
            topRight,
            bottomRight,
        ];
    }

    private static bool AllPointsInside(Point2f[] points, int width, int height) =>
        points.All(point =>
            point.X >= 0 &&
            point.Y >= 0 &&
            point.X < width &&
            point.Y < height);

    private static float Distance(Point2f first, Point2f second)
    {
        float x = first.X - second.X;
        float y = first.Y - second.Y;
        return MathF.Sqrt(x * x + y * y);
    }

}
