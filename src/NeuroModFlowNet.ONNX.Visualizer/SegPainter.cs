namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Utility class for visualizing Segmentation detection results.
/// RU: Утилитарный класс для визуализации результатов детекции сегментации.
/// </summary>
public static class SegPainter
{
    /// <summary>
    /// EN: Draws Segmentation detections with computed pixel masks (from YoloSeg32Result).
    /// RU: Рисует результаты сегментации с масками (из YoloSeg32Result).
    /// </summary>
    public static void DrawSeg(
        Mat mat,
        YoloSeg_FP32_XYWHSC_Mask32[] boxes,
        float[][] masks,
        LetterboxInfo info,
        float scaleX, float scaleY,
        Func<int, string>? nameResolver = null)
    {
        using Mat overlay = mat.Clone();

        for (int d = 0; d < boxes.Length; d++)
        {
            var box   = boxes[d];
            var color = VisualUtils.ClassColor((int)box.ClassId);

            // ── Bounding box coordinates ────────────────────────────────
            var p1 = info.MapBack(box.X, box.Y);
            var p2 = info.MapBack(box.W, box.H);

            int x1 = (int)(p1.X * scaleX);
            int y1 = (int)(p1.Y * scaleY);
            int x2 = (int)(p2.X * scaleX);
            int y2 = (int)(p2.Y * scaleY);

            var r = new Rect(
                Math.Max(0, x1),
                Math.Max(0, y1),
                Math.Clamp(x2 - x1, 1, mat.Width  - Math.Max(0, x1)),
                Math.Clamp(y2 - y1, 1, mat.Height - Math.Max(0, y1)));

            // ── Pixel mask processing ──────────────────────────────────
            if (d < masks.Length && masks[d].Length > 0)
            {
                int mLen  = masks[d].Length;
                int mSide = (int)Math.Sqrt(mLen);

                // 1. Create source mask Mat (e.g. 160x160)
                using var maskMat = new Mat(mSide, mSide, MatType.CV_32FC1);
                System.Runtime.InteropServices.Marshal.Copy(masks[d], 0, maskMat.Data, mLen);

                // 2. Scale mask to the MODEL'S SQUARE INPUT SIZE
                using var modelSpaceMask = new Mat();
                Cv2.Resize(maskMat, modelSpaceMask, new Size(info.TargetWidth, info.TargetHeight));

                // 3. Crop out the Letterbox padding
                int imgW = info.TargetWidth  - 2 * info.OffsetX;
                int imgH = info.TargetHeight - 2 * info.OffsetY;
                var cropRect = new Rect(info.OffsetX, info.OffsetY, imgW, imgH);
                
                using var actualImgMask = modelSpaceMask[cropRect];

                // 4. Resize the "clean" image mask portion to the actual VIEW cell size
                using var fullFrameMask = new Mat();
                Cv2.Resize(actualImgMask, fullFrameMask, new Size(mat.Width, mat.Height));

                // 5. Threshold and convert to 8U
                using var maskBin = new Mat();
                Cv2.Threshold(fullFrameMask, maskBin, 0.5, 1.0, ThresholdTypes.Binary);
                using var mask8u = new Mat();
                maskBin.ConvertTo(mask8u, MatType.CV_8UC1, 255.0);

                // 6. Crop by bounding box and draw
                using var bboxMask = new Mat(mat.Size(), MatType.CV_8UC1, Scalar.Black);
                Cv2.Rectangle(bboxMask, r, Scalar.White, -1);
                
                using var finalMask = new Mat();
                Cv2.BitwiseAnd(mask8u, bboxMask, finalMask);

                using var colorMat = new Mat(mat.Size(), MatType.CV_8UC3, color);
                colorMat.CopyTo(overlay, finalMask);
            }
            else if (r.Width > 2 && r.Height > 2)
            {
                overlay[r].SetTo(color);
            }

            Cv2.Rectangle(mat, r, color, 2, LineTypes.AntiAlias);
            string name = nameResolver?.Invoke((int)box.ClassId) ?? $"#{(int)box.ClassId}";
            VisualUtils.DrawLabel(mat, $"{name} {box.Score:P0}", r.TopLeft, color);
        }

        Cv2.AddWeighted(overlay, 0.35, mat, 0.65, 0, mat);
    }

    /// <summary>
    /// EN: Legacy overload for Seg_single (no pixel masks, bbox fill only).
    /// RU: Устаревший метод для Seg_single (без масок, только заполнение бокса).
    /// </summary>
    public static void DrawSeg(
        Mat mat,
        YoloSeg_FP32[] boxes,
        LetterboxInfo info,
        float scaleX, float scaleY,
        Func<int, string>? nameResolver = null)
    {
        using Mat overlay = mat.Clone();

        foreach (var box in boxes)
        {
            var p1 = info.MapBack(box.X, box.Y);
            var p2 = info.MapBack(box.W, box.H);

            int x1 = (int)(p1.X * scaleX);
            int y1 = (int)(p1.Y * scaleY);
            int x2 = (int)(p2.X * scaleX);
            int y2 = (int)(p2.Y * scaleY);

            var color = VisualUtils.ClassColor((int)box.ClassId);
            var r = new Rect(
                Math.Max(0, x1),
                Math.Max(0, y1),
                Math.Min(mat.Width  - Math.Max(0, x1), x2 - x1),
                Math.Min(mat.Height - Math.Max(0, y1), y2 - y1));

            if (r.Width > 2 && r.Height > 2)
                overlay[r].SetTo(color);

            Cv2.Rectangle(mat, r, color, 2, LineTypes.AntiAlias);
            string name = nameResolver?.Invoke((int)box.ClassId) ?? $"#{(int)box.ClassId}";
            VisualUtils.DrawLabel(mat, $"{name} {box.Score:P0}", r.TopLeft, color);
        }

        Cv2.AddWeighted(overlay, 0.30, mat, 0.70, 0, mat);
    }
}
