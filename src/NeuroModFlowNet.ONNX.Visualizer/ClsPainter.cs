namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Utility class for visualizing Classification results.
/// RU: Утилитарный класс для визуализации результатов классификации.
/// </summary>
public static class ClsPainter
{
    /// <summary>
    /// EN: Draws Classification result as a centered overlay with class id and score bar.
    /// RU: Рисует результат классификации в виде оверлея с идентификатором класса и шкалой вероятности.
    /// </summary>
    public static void DrawCls(Mat mat, YoloCls result, Func<int, string>? nameResolver = null)
    {
        // Background tint
        using var overlay = mat.Clone();
        Cv2.Rectangle(overlay, new Rect(0, 0, mat.Width, mat.Height), new Scalar(20, 20, 40), -1);
        Cv2.AddWeighted(overlay, 0.45, mat, 0.55, 0, mat);

        string name  = nameResolver?.Invoke(result.ClassId) ?? $"Class #{result.ClassId}";
        string score = $"{result.Score:P1}";

        // Large centered class name
        double fontSize = 1.2;
        int    thick    = 2;
        var    tsz      = Cv2.GetTextSize(name, HersheyFonts.HersheySimplex, fontSize, thick, out _);
        var    pt       = new Point((mat.Width - tsz.Width) / 2, mat.Height / 2 - 10);
        Cv2.PutText(mat, name, pt, HersheyFonts.HersheySimplex, fontSize,
            new Scalar(50, 220, 255), thick, LineTypes.AntiAlias);

        // Score below
        var scoreSz = Cv2.GetTextSize(score, HersheyFonts.HersheySimplex, 0.8, 1, out _);
        var scorePt = new Point((mat.Width - scoreSz.Width) / 2, mat.Height / 2 + 35);
        Cv2.PutText(mat, score, scorePt, HersheyFonts.HersheySimplex, 0.8,
            new Scalar(100, 255, 150), 1, LineTypes.AntiAlias);

        // Score bar
        int barW   = (int)(mat.Width * 0.7f * result.Score);
        int barX   = (int)(mat.Width * 0.15f);
        int barY   = mat.Height / 2 + 60;
        Cv2.Rectangle(mat, new Rect(barX, barY, (int)(mat.Width * 0.7f), 12), new Scalar(50, 50, 50), -1);
        Cv2.Rectangle(mat, new Rect(barX, barY, barW, 12), new Scalar(50, 220, 100), -1);

        // Top-3 labels
        var top3 = result.Scores
            .Select((s, i) => (Score: s, Id: i))
            .OrderByDescending(x => x.Score)
            .Take(3)
            .ToArray();

        for (int t = 0; t < top3.Length; t++)
        {
            string clsName = nameResolver?.Invoke(top3[t].Id) ?? $"#{top3[t].Id}";
            string lbl = $"{clsName}: {top3[t].Score:P0}";
            Cv2.PutText(mat, lbl, new Point(10, mat.Height - 50 + t * 20),
                HersheyFonts.HersheySimplex, 0.45, new Scalar(180, 180, 180), 1, LineTypes.AntiAlias);
        }
    }
}
