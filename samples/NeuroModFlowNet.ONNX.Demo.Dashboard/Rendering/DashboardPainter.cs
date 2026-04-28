using System;
using System.Linq;
using OpenCvSharp;
using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

/// <summary>
/// Dashboard renderer: 6-cell grid (3 cols x 2 rows), one cell per model type.
/// Row 0: Raw Camera | Box Detection | OBB Detection
/// Row 1: Segmentation | Pose Estimation | Classification
/// </summary>
internal class DashboardPainter : IDisposable
{
    private readonly int _width;
    private readonly int _height;
    private readonly Mat _canvas;
    private bool _disposed;

    // ---------- COCO 80-class palette (hue-stepped) ----------
    private static readonly Scalar[] ClassColors = Enumerable
        .Range(0, 80)
        .Select(i => HsvToScalar(i * 4.5f, 220, 200))
        .ToArray();

    // ---------- COCO 17-keypoint skeleton ----------
    private static readonly (int A, int B)[] PoseSkeleton =
    [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ];

    public DashboardPainter(int width = 1920, int height = 1080)
    {
        _width = width;
        _height = height;
        _canvas = new Mat(height, width, MatType.CV_8UC3, Scalar.Black);
    }

    // =====================================================================
    //  Main Draw entry point
    // =====================================================================

    /// <summary>
    /// Renders all 6 cells and returns the composited canvas Mat.
    /// Each entry in <paramref name="views"/> is (Title, DrawAction).
    /// DrawAction receives a resized copy of <paramref name="sourceFrame"/>
    /// and should annotate it in-place.
    /// </summary>
    public Mat Draw(
        Mat sourceFrame,
        IReadOnlyList<DashboardView> views,
        double totalFps,
        double inferenceMs = 0)
    {
        _canvas.SetTo(Scalar.Black);

        const int rows = 2;
        const int cols = 3;

        int statusBarH = 36;
        int usableH = _height - statusBarH;

        int cellW = _width / cols;
        int cellH = usableH / rows;

        for (int i = 0; i < views.Count && i < rows * cols; i++)
        {
            int row = i / cols;
            int col = i % cols;

            var roi = new Rect(col * cellW, row * cellH, cellW, cellH);
            using Mat cell = _canvas[roi];

            using Mat resized = new Mat();
            Cv2.Resize(sourceFrame, resized, new Size(cellW, cellH));

            try { views[i].DrawAction(resized); }
            catch { /* never crash the loop */ }

            // Title badge
            DrawTitleBadge(resized, views[i].Title, col, row);

            resized.CopyTo(cell);

            // Cell border
            Cv2.Rectangle(_canvas, roi, new Scalar(60, 60, 60), 1);
        }

        // ── Status bar ──────────────────────────────────────────────────
        string infInfo = inferenceMs > 0 ? $" |  INF: {inferenceMs,5:F1} ms ({1000.0 / inferenceMs,4:F0} FPS)" : "";
        string status = $"  TOTAL FPS: {totalFps,4:F1}{infInfo}  |  GPU: TensorRT  |  NeuroModFlowNet.ONNX";

        Cv2.Rectangle(_canvas, new Rect(0, _height - statusBarH, _width, statusBarH), new Scalar(18, 18, 18), -1);
        Cv2.PutText(_canvas, status, new Point(10, _height - 10),
            HersheyFonts.HersheySimplex, 0.55, new Scalar(180, 180, 180), 1, LineTypes.AntiAlias);

        return _canvas;
    }

    // =====================================================================
    //  Private helpers
    // =====================================================================

    private static Scalar HsvToScalar(float h, float s, float v) => VisualUtils.HsvToScalar(h, s, v);

    private static void DrawTitleBadge(Mat mat, string title, int col, int row)
    {
        // Color-coded badge per model type
        Scalar badgeColor = (col, row) switch
        {
            (0, 0) => new Scalar(30, 30, 30),        // Raw
            (1, 0) => new Scalar(0, 60, 0),          // Box  — green tint
            (2, 0) => new Scalar(50, 30, 0),          // OBB  — amber tint
            (0, 1) => new Scalar(0, 30, 60),         // Seg  — blue tint
            (1, 1) => new Scalar(50, 0, 0),          // Pose — red tint
            (2, 1) => new Scalar(40, 0, 40),         // Cls  — purple tint
            _ => new Scalar(30, 30, 30)
        };

        var tsz = Cv2.GetTextSize(title, HersheyFonts.HersheySimplex, 0.55, 1, out _);
        Cv2.Rectangle(mat, new Rect(0, 0, tsz.Width + 16, 30), badgeColor, -1);
        Cv2.PutText(mat, title, new Point(8, 22),
            HersheyFonts.HersheySimplex, 0.55, Scalar.White, 1, LineTypes.AntiAlias);
    }

    public void Dispose()
    {
        if (!_disposed) { _canvas.Dispose(); _disposed = true; }
    }
}
