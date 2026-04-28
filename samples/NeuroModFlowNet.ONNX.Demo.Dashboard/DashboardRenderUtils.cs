using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal static class DashboardRenderUtils
{
    public static void DrawNoModel(Mat mat, string message)
    {
        using var overlay = mat.Clone();
        Cv2.Rectangle(overlay, new Rect(0, 0, mat.Width, mat.Height), new Scalar(20, 20, 20), -1);
        Cv2.AddWeighted(overlay, 0.6, mat, 0.4, 0, mat);

        var size = Cv2.GetTextSize(message, HersheyFonts.HersheySimplex, 0.6, 1, out _);
        Cv2.PutText(
            mat,
            message,
            new Point((mat.Width - size.Width) / 2, mat.Height / 2),
            HersheyFonts.HersheySimplex,
            0.6,
            new Scalar(100, 100, 100),
            1,
            LineTypes.AntiAlias);
    }
}
