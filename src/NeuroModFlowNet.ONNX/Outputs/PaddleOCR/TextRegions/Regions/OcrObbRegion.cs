using OpenCvSharp;
using System.Runtime.InteropServices;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Lightweight oriented text region used before it is expanded to four corner points.
/// RU: Легкая ориентированная текстовая область до разворачивания в четыре угловые точки.
/// </summary>
/// <remarks>
/// EN: OBB is useful at the detector boundary because YOLO emits center, size, and angle. The extraction
/// boundary still switches to <see cref="OcrQuadRegion"/> so all later geometry is point-based and independent
/// from detector-specific box formats.
/// RU: OBB удобен на границе детектора, потому что YOLO отдает центр, размер и угол. Граница вырезания все равно
/// переходит на <see cref="OcrQuadRegion"/>, чтобы дальнейшая геометрия работала через точки и не зависела
/// от формата боксов конкретного детектора.
/// </remarks>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly record struct OcrObbRegion(
    float CenterX,
    float CenterY,
    float Width,
    float Height,
    float AngleRadians)
{
    public void GetPoints(Span<Point2f> points)
    {
        if(points.Length < 4)
            throw new ArgumentException("Destination must contain at least four points.", nameof(points));

        float halfWidth = Width * 0.5f;
        float halfHeight = Height * 0.5f;
        float cos = MathF.Cos(AngleRadians);
        float sin = MathF.Sin(AngleRadians);

        WriteRotatedPoint(points, 0, -halfWidth, -halfHeight, cos, sin);
        WriteRotatedPoint(points, 1,  halfWidth, -halfHeight, cos, sin);
        WriteRotatedPoint(points, 2,  halfWidth,  halfHeight, cos, sin);
        WriteRotatedPoint(points, 3, -halfWidth,  halfHeight, cos, sin);
    }

    private void WriteRotatedPoint(
        Span<Point2f> points,
        int index,
        float localX,
        float localY,
        float cos,
        float sin)
    {
        points[index] = new Point2f(
            CenterX + localX * cos - localY * sin,
            CenterY + localX * sin + localY * cos);
    }
}
