using System.Diagnostics;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace OnnxTestLoader;

public sealed class FpsConsoleMonitor
{
    private static readonly char[] AsciiSparklineChars = ['.', ':', '-', '=', '+', '*', '#', '@'];

    private readonly Queue<double> values = new();
    private readonly Stopwatch renderStopwatch = Stopwatch.StartNew();
    private readonly int capacity;
    private readonly TimeSpan updateInterval;
    private readonly FpsTrendMode trendMode;
    private readonly FpsTrendShape trendShape;
    private readonly int trendHeight;

    private double current;

    public FpsConsoleMonitor(
        int capacity = 60,
        TimeSpan? updateInterval = null,
        FpsTrendMode trendMode = FpsTrendMode.Braille,
        FpsTrendShape trendShape = FpsTrendShape.Bar,
        int trendHeight = 5)
    {
        this.capacity = capacity;
        this.updateInterval = updateInterval ?? TimeSpan.FromMilliseconds(300);
        this.trendMode = trendMode;
        this.trendShape = trendShape;
        this.trendHeight = Math.Clamp(trendHeight, 4, 6);
    }

    public bool ShouldRender => renderStopwatch.Elapsed >= updateInterval;

    public void AddFrame(double fps)
    {
        if(double.IsNaN(fps) || double.IsInfinity(fps) || fps <= 0)
            return;

        current = fps;
        values.Enqueue(fps);

        while(values.Count > capacity)
            values.Dequeue();
    }

    public Panel Render()
    {
        double average = values.Count == 0 ? 0 : values.Average();
        double min = values.Count == 0 ? 0 : values.Min();
        double max = values.Count == 0 ? 0 : values.Max();

        var metricsGrid = new Grid();
        metricsGrid.AddColumn(new GridColumn().NoWrap());
        metricsGrid.AddColumn(new GridColumn().NoWrap());
        metricsGrid.AddRow(new Markup("[grey]Current[/]"), new Markup($"[bold cyan]{current:F1}[/]"));
        metricsGrid.AddRow(new Markup("[grey]Average[/]"), new Markup($"[green]{average:F1}[/]"));
        metricsGrid.AddRow(new Markup("[grey]Min / Max[/]"), new Markup($"[yellow]{min:F1}[/] [grey]/[/] [yellow]{max:F1}[/]"));

        var rows = new Rows(
            metricsGrid,
            new Markup($"[grey]Trend {trendShape}[/]"),
            CreateTrend(min, max));

        return new Panel(rows)
            .Header("[cyan]FPS[/]")
            .Border(BoxBorder.Rounded);
    }

    public void RestartRenderTimer() => renderStopwatch.Restart();

    private IRenderable CreateTrend(double min, double max) =>
        trendMode switch
        {
            FpsTrendMode.Canvas => CreateTrendCanvas(),
            FpsTrendMode.Braille => CreateBrailleTrend(min, max),
            FpsTrendMode.Ascii => CreateAsciiTrend(min, max),
            _ => CreateTrendCanvas(),
        };

    private Canvas CreateTrendCanvas()
    {
        int height = trendHeight * 2;

        int width = Math.Max(1, capacity);
        var canvas = new Canvas(width, height)
        {
            Scale = false,
        };

        if(values.Count == 0)
            return canvas;

        double min = values.Min();
        double max = values.Max();
        double range = max - min;

        double[] snapshot = values.ToArray();
        int offsetX = Math.Max(0, width - snapshot.Length);

        for(int x = 0; x < snapshot.Length; x++)
        {
            double normalized = range <= double.Epsilon
                ? 1
                : (snapshot[x] - min) / range;

            int barHeight = Math.Clamp((int)Math.Round(normalized * (height - 1)) + 1, 1, height);
            int canvasX = offsetX + x;

            for(int y = height - 1; y >= height - barHeight; y--)
                canvas.SetPixel(canvasX, y, Color.Cyan1);
        }

        return canvas;
    }

    private Markup CreateBrailleTrend(double min, double max)
    {
        if(values.Count == 0)
            return new Markup("[grey]waiting...[/]");

        double[] snapshot = values.ToArray();
        double range = max - min;

        int width = (snapshot.Length + 1) / 2;
        int dotHeight = trendHeight * 4;
        bool[,] pixels = new bool[dotHeight, width * 2];

        for(int index = 0; index < snapshot.Length; index++)
        {
            int x = index;
            int normalizedY = GetTrendY(snapshot[index], min, range, dotHeight);

            if(trendShape == FpsTrendShape.Line)
            {
                pixels[normalizedY, x] = true;
            }
            else
            {
                for(int y = normalizedY; y < dotHeight; y++)
                    pixels[y, x] = true;
            }
        }

        string[] lines = new string[trendHeight];
        for(int row = 0; row < trendHeight; row++)
        {
            char[] chars = new char[width];
            for(int column = 0; column < width; column++)
                chars[column] = CreateBrailleCell(pixels, row, column);

            string label = row == 0
                ? $" [yellow]{max:F1}[/]"
                : row == trendHeight - 1
                    ? $" [yellow]{min:F1}[/]"
                    : string.Empty;

            lines[row] = $"[cyan]{Markup.Escape(new string(chars))}[/]{label}";
        }

        return new Markup(string.Join(Environment.NewLine, lines));
    }

    private Markup CreateAsciiTrend(double min, double max)
    {
        if(values.Count == 0)
            return new Markup("[grey]waiting...[/]");

        double[] snapshot = values.ToArray();
        double range = max - min;
        char[,] chars = new char[trendHeight, snapshot.Length];

        for(int row = 0; row < trendHeight; row++)
        {
            for(int column = 0; column < snapshot.Length; column++)
                chars[row, column] = ' ';
        }

        for(int x = 0; x < snapshot.Length; x++)
        {
            int y = GetTrendY(snapshot[x], min, range, trendHeight);

            if(trendShape == FpsTrendShape.Line)
            {
                chars[y, x] = '*';
            }
            else
            {
                for(int row = y; row < trendHeight; row++)
                    chars[row, x] = AsciiSparklineChars[^1];
            }
        }

        string[] lines = new string[trendHeight];
        for(int row = 0; row < trendHeight; row++)
        {
            string label = row == 0
                ? $" [yellow]{max:F1}[/]"
                : row == trendHeight - 1
                    ? $" [yellow]{min:F1}[/]"
                    : string.Empty;

            lines[row] = $"[cyan]{Markup.Escape(new string(GetRow(chars, row)))}[/]{label}";
        }

        return new Markup(string.Join(Environment.NewLine, lines));
    }

    private static int GetTrendY(double value, double min, double range, int height)
    {
        if(range <= double.Epsilon)
            return height / 2;

        double normalized = (value - min) / range;
        int y = (height - 1) - (int)Math.Round(normalized * (height - 1));
        return Math.Clamp(y, 0, height - 1);
    }

    private static char CreateBrailleCell(bool[,] pixels, int row, int column)
    {
        int mask = 0;
        int baseY = row * 4;
        int baseX = column * 2;

        if(GetPixel(pixels, baseY + 0, baseX + 0)) mask |= 0x01;
        if(GetPixel(pixels, baseY + 1, baseX + 0)) mask |= 0x02;
        if(GetPixel(pixels, baseY + 2, baseX + 0)) mask |= 0x04;
        if(GetPixel(pixels, baseY + 3, baseX + 0)) mask |= 0x40;
        if(GetPixel(pixels, baseY + 0, baseX + 1)) mask |= 0x08;
        if(GetPixel(pixels, baseY + 1, baseX + 1)) mask |= 0x10;
        if(GetPixel(pixels, baseY + 2, baseX + 1)) mask |= 0x20;
        if(GetPixel(pixels, baseY + 3, baseX + 1)) mask |= 0x80;

        return (char)(0x2800 + mask);
    }

    private static bool GetPixel(bool[,] pixels, int y, int x) =>
        y >= 0
        && y < pixels.GetLength(0)
        && x >= 0
        && x < pixels.GetLength(1)
        && pixels[y, x];

    private static char[] GetRow(char[,] chars, int row)
    {
        char[] result = new char[chars.GetLength(1)];
        for(int column = 0; column < result.Length; column++)
            result[column] = chars[row, column];

        return result;
    }
}
