namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Utility class for visualizing Pose Estimation results (keypoints and skeleton).
/// RU: Утилитарный класс для визуализации результатов оценки позы (ключевые точки и скелет).
/// </summary>
public static class YoloPosePainter
{
    // ---------- COCO 17-keypoint skeleton ----------
    public static readonly (int A, int B)[] PoseSkeleton =
    [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ];

    public static void DrawPose<T>(
        Mat mat,
        T[] detections,
        LetterboxInfo info,
        float scaleX, float scaleY,
        Func<int, string>? nameResolver = null) where T : struct, IOutAsT<YoloPose>
    {
        foreach (var item in detections)
        {
            var det = item.AsStd();

            var p1 = info.MapBack(det.X, det.Y);
            var p2 = info.MapBack(det.W, det.H);

            int x1 = (int)(p1.X * scaleX);
            int y1 = (int)(p1.Y * scaleY);
            int x2 = (int)(p2.X * scaleX);
            int y2 = (int)(p2.Y * scaleY);

            var boxColor = new Scalar(0, 200, 255);
            var r = new Rect(x1, y1, x2 - x1, y2 - y1);
            Cv2.Rectangle(mat, r, boxColor, 2, LineTypes.AntiAlias);

            string name = nameResolver?.Invoke(0) ?? ""; 
            string lbl  = string.IsNullOrEmpty(name) ? $"{det.Score:P0}" : $"{name} {det.Score:P0}";
            VisualUtils.DrawLabel(mat, lbl, r.TopLeft, boxColor);

            var kpts = det.Keypoints;
            int numKpts = kpts.Length;
            if (numKpts == 0) continue;

            // Build point array
            var pts = new Point[numKpts];

            for (int k = 0; k < numKpts; k++)
            {
                var kOrig = info.MapBack(kpts[k].X, kpts[k].Y);
                pts[k] = new Point((int)(kOrig.X * scaleX), (int)(kOrig.Y * scaleY));
            }

            // Skeleton (Only for 17 kpts COCO)
            if (numKpts == 17)
            {
                foreach (var (a, b) in PoseSkeleton)
                {
                    Cv2.Line(mat, pts[a], pts[b], new Scalar(0, 255, 128), 2, LineTypes.AntiAlias);
                }
            }

            // Keypoint dots
            for (int k = 0; k < numKpts; k++)
            {
                Cv2.Circle(mat, pts[k], 4, new Scalar(255, 80, 80), -1, LineTypes.AntiAlias);
            }
        }
    }
}
