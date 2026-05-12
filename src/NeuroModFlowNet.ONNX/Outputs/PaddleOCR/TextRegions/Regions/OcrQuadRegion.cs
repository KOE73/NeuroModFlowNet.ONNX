using System.Runtime.InteropServices;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Four source-image points describing one text region prepared for OCR recognition extraction.
/// RU: Четыре точки исходного изображения, описывающие одну текстовую область для OCR-вырезания.
/// </summary>
/// <remarks>
/// EN: The layout is intentionally flat and sequential: x0,y0,x1,y1,x2,y2,x3,y3.
/// Batch code can reinterpret a span of regions as floats when it needs a [N,2] coordinate view.
/// Four points are the common denominator for OBB boxes, perspective crops, column analysis, and future
/// neural text-region predictors; angle and size are derived data and should not be the extractor contract.
/// RU: Layout намеренно плоский и последовательный: x0,y0,x1,y1,x2,y2,x3,y3.
/// Batch-код может переинтерпретировать span регионов как float, когда нужен координатный вид [N,2].
/// Четыре точки являются общим форматом для OBB-боксов, perspective crop, анализа колонок и будущих
/// нейросетевых выделителей текста; угол и размер являются вычисляемыми данными и не должны быть контрактом вырезалки.
/// </remarks>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly record struct OcrQuadRegion(
    float X0,
    float Y0,
    float X1,
    float Y1,
    float X2,
    float Y2,
    float X3,
    float Y3)
{
    public Point2f Point0 => new(X0, Y0);
    public Point2f Point1 => new(X1, Y1);
    public Point2f Point2 => new(X2, Y2);
    public Point2f Point3 => new(X3, Y3);

    public static OcrQuadRegion FromPoints(ReadOnlySpan<Point2f> points)
    {
        if(points.Length < 4)
            throw new ArgumentException("At least four points are required.", nameof(points));

        return new OcrQuadRegion(
            points[0].X,
            points[0].Y,
            points[1].X,
            points[1].Y,
            points[2].X,
            points[2].Y,
            points[3].X,
            points[3].Y);
    }

    public void CopyTo(Span<Point2f> points)
    {
        if(points.Length < 4)
            throw new ArgumentException("Destination must contain at least four points.", nameof(points));

        points[0] = Point0;
        points[1] = Point1;
        points[2] = Point2;
        points[3] = Point3;
    }
}
