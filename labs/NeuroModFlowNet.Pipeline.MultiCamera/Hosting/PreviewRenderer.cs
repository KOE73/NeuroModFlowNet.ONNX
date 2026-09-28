using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Lab-only overlay on the VM preview Mat: zones, raw detections, tracks with trails, text regions and OCR strings.
/// Coordinates come in rect space and are scaled to the preview size.
///
/// RU: Оверлей только для лаборатории поверх превью из VM: зоны, детекции, треки с траекторией, текстовые регионы и
/// строки OCR. Координаты приходят в rect-пространстве и масштабируются под размер превью.
/// </summary>
internal static class PreviewRenderer
{
    static readonly Scalar StartZoneColor = new(0, 200, 0);
    static readonly Scalar EndZoneColor = new(0, 140, 255);
    static readonly Scalar DetectionColor = new(255, 200, 0);
    static readonly Scalar PendingTrackColor = new(0, 0, 255);
    static readonly Scalar ConfirmedTrackColor = new(0, 255, 255);
    static readonly Scalar TextRegionColor = new(255, 0, 255);
    static readonly Scalar HeaderColor = new(255, 255, 255);

    public static void Draw(
        Mat preview,
        CameraGeometry geometry,
        CameraConfig camera,
        TrackDetection[] detections,
        TrackedObject[] tracks,
        YoloObb[] textRegions,
        int[] textTrackIds,
        PaddleOCRRecExtractor.OcrResult[] recognized,
        CameraStatisticsSnapshot statistics,
        double potentialFps)
    {
        double scale = preview.Width / (double)geometry.Width;

        if(camera.Tracking?.StartZone is { } startZone)
            DrawZone(preview, startZone, scale, StartZoneColor, "start");

        if(camera.Tracking?.EndZone is { } endZone)
            DrawZone(preview, endZone, scale, EndZoneColor, "end");

        foreach(TrackDetection detection in detections)
        {
            var box = new YoloObb { X = detection.X, Y = detection.Y, W = detection.W, H = detection.H, Angle = detection.Angle };
            DrawObb(preview, box, scale, DetectionColor, 1);
            Cv2.PutText(preview, $"c{detection.ClassId} {detection.Score:F2}", ScalePoint(detection.X - (detection.W / 2), detection.Y + (detection.H / 2) + 12, scale), HersheyFonts.HersheySimplex, 0.4, DetectionColor, 1);
        }

        foreach(TrackedObject track in tracks)
        {
            Scalar color = track.IsConfirmed ? ConfirmedTrackColor : PendingTrackColor;
            var box = new YoloObb { X = track.X, Y = track.Y, W = track.W, H = track.H, Angle = track.Angle };
            DrawObb(preview, box, scale, color, track.IsConfirmed ? 2 : 1);

            for(int index = 1; index < track.Path.Count; index++)
            {
                Cv2.Line(
                    preview,
                    ScalePoint(track.Path[index - 1].X, track.Path[index - 1].Y, scale),
                    ScalePoint(track.Path[index].X, track.Path[index].Y, scale),
                    color,
                    1);
            }

            string label = track.IsConfirmed ? $"T{track.ConfirmedTrackId} age {track.Age}" : $"t{track.TrackId} {track.Age}";
            Cv2.PutText(preview, label, ScalePoint(track.X - (track.W / 2), track.Y - (track.H / 2) - 4, scale), HersheyFonts.HersheySimplex, 0.45, color, 1);
        }

        for(int index = 0; index < textRegions.Length; index++)
        {
            DrawObb(preview, textRegions[index], scale, TextRegionColor, 1);
            if(index < recognized.Length && !string.IsNullOrWhiteSpace(recognized[index].Standard))
            {
                string trackPrefix = index < textTrackIds.Length ? $"T{textTrackIds[index]} " : string.Empty;
                Cv2.PutText(
                    preview,
                    trackPrefix + recognized[index].Standard,
                    ScalePoint(textRegions[index].X, textRegions[index].Y, scale),
                    HersheyFonts.HersheySimplex,
                    0.5,
                    TextRegionColor,
                    1);
            }
        }

        string header = $"{camera.Id} run {statistics.RunsCompleted} | {statistics.AverageFps:F1} fps (max {potentialFps:F0}) | {statistics.AverageRunMilliseconds:F1} ms | det {detections.Length} trk {tracks.Length}";
        Cv2.PutText(preview, header, new Point(6, 18), HersheyFonts.HersheySimplex, 0.5, HeaderColor, 1);
    }

    static void DrawZone(Mat preview, ZoneConfig zone, double scale, Scalar color, string label)
    {
        Point topLeft = ScalePoint(zone.X1, zone.Y1, scale);
        Point bottomRight = ScalePoint(zone.X2, zone.Y2, scale);
        Cv2.Rectangle(preview, topLeft, bottomRight, color, 1);
        Cv2.PutText(preview, label, new Point(topLeft.X + 4, topLeft.Y + 14), HersheyFonts.HersheySimplex, 0.45, color, 1);
    }

    static void DrawObb(Mat preview, in YoloObb box, double scale, Scalar color, int thickness)
    {
        var rotated = new RotatedRect(
            new Point2f((float)(box.X * scale), (float)(box.Y * scale)),
            new Size2f((float)(box.W * scale), (float)(box.H * scale)),
            box.Angle * 180f / MathF.PI);

        Point[] corners = rotated.Points().Select(static point => new Point((int)MathF.Round(point.X), (int)MathF.Round(point.Y))).ToArray();
        Cv2.Polylines(preview, [corners], isClosed: true, color, thickness);
    }

    static Point ScalePoint(float x, float y, double scale) =>
        new((int)Math.Round(x * scale), (int)Math.Round(y * scale));
}
