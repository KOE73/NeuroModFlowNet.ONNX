using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using OpenCvSharp;
using Spectre.Console;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX.Lab.Algorithms;

internal static class Program
{
    private const int DefaultWidth = 640;
    private const int DefaultHeight = 640;
    private const int DefaultBatch = 4;
    private const int PreviewCount = 12;

    static void Main(string[] args)
    {
        int width = GetArg(args, 0, DefaultWidth);
        int height = GetArg(args, 1, DefaultHeight);
        int batch = GetArg(args, 2, DefaultBatch);

        AnsiConsole.Write(new FigletText("Algorithms Lab").Color(Color.Cyan1));
        AnsiConsole.MarkupLine($"[grey]Image:[/] {width}x{height}  [grey]Batch:[/] {batch}");
        AnsiConsole.WriteLine();

        using Mat sourceMat = PositiveAlgorithmHelpers.CreateFastRealMat(width, height);
        PrintMatInfo(sourceMat);

        RunPositiveHelpers(sourceMat, width, height, batch);
        RunSymmetricHelpers(sourceMat, width, height, batch);

        AnsiConsole.MarkupLine("[green]Done.[/]");
    }

    private static void RunPositiveHelpers(Mat sourceMat, int width, int height, int batch)
    {
        AnsiConsole.Write(new Rule("[yellow]PositiveAlgorithmHelpers[/]").RuleStyle("grey"));

        RunCheck("Positive.CheckAlgorithms()", log => PositiveAlgorithmHelpers.CheckAlgorithms(log));
        RunCheck("Positive.CheckAlgorithms(Mat, batch)", log => PositiveAlgorithmHelpers.CheckAlgorithms(sourceMat, batch, log));

        RunMatFactoryPreview("Positive.CreateFastRealMat(width, height)", width, height);
        RunCloneBatchPreview("Positive.CloneMatBatch(Mat, batch)", sourceMat, batch);

        InputDataToSpanBufConverter<List<Mat>, Float16> byMat =
            RunAutoTune("Positive.AutoTuneFP16(Mat, batch)", log => PositiveAlgorithmHelpers.AutoTuneFP16(sourceMat, batch, log));
        PreviewFP16("Positive.AutoTuneFP16(Mat, batch) selected delegate output", sourceMat, batch, byMat);

        InputDataToSpanBufConverter<List<Mat>, Float16> bySize =
            RunAutoTune("Positive.AutoTuneFP16(width, height, batch)", log => PositiveAlgorithmHelpers.AutoTuneFP16(width, height, batch, log));
        PreviewFP16("Positive.AutoTuneFP16(width, height, batch) selected delegate output", sourceMat, batch, bySize);
    }

    private static void RunSymmetricHelpers(Mat sourceMat, int width, int height, int batch)
    {
        AnsiConsole.Write(new Rule("[yellow]SymmetricAlgorithmHelpers[/]").RuleStyle("grey"));

        RunCheck("Symmetric.CheckAlgorithms()", log => SymmetricAlgorithmHelpers.CheckAlgorithms(log));
        RunCheck("Symmetric.CheckAlgorithms(Mat, batch)", log => SymmetricAlgorithmHelpers.CheckAlgorithms(sourceMat, batch, log));

        InputDataToSpanBufConverter<List<Mat>, float> fp32 =
            RunAutoTune("Symmetric.AutoTuneFP32(Mat, batch)", log => SymmetricAlgorithmHelpers.AutoTuneFP32(sourceMat, batch, log));
        PreviewFP32("Symmetric.AutoTuneFP32(Mat, batch) selected delegate output", sourceMat, batch, fp32);

        InputDataToSpanBufConverter<List<Mat>, Float16> fp16 =
            RunAutoTune("Symmetric.AutoTuneFP16(Mat, batch)", log => SymmetricAlgorithmHelpers.AutoTuneFP16(sourceMat, batch, log));
        PreviewFP16("Symmetric.AutoTuneFP16(Mat, batch) selected delegate output", sourceMat, batch, fp16);
    }

    private static void RunCheck(string title, Func<Action<string>, bool> action)
    {
        AlgorithmLog log = new();
        Stopwatch stopwatch = Stopwatch.StartNew();
        bool result = action(log.Add);
        stopwatch.Stop();

        Table table = new Table()
            .Title(title)
            .AddColumn("Result")
            .AddColumn("Elapsed");

        table.AddRow(result ? "[green]OK[/]" : "[red]ERROR[/]", $"{stopwatch.Elapsed.TotalMilliseconds:F3} ms");
        AnsiConsole.Write(table);
        log.Write(title);
    }

    private static T RunAutoTune<T>(string title, Func<Action<string>, T> action)
    {
        AlgorithmLog log = new();
        Stopwatch stopwatch = Stopwatch.StartNew();
        T result = action(log.Add);
        stopwatch.Stop();

        Table table = new Table()
            .Title(title)
            .AddColumn("Returned")
            .AddColumn("Elapsed");

        table.AddRow(typeof(T).Name, $"{stopwatch.Elapsed.TotalMilliseconds:F3} ms");
        AnsiConsole.Write(table);
        log.Write(title);
        return result;
    }

    private static void RunMatFactoryPreview(string title, int width, int height)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        using Mat mat = PositiveAlgorithmHelpers.CreateFastRealMat(width, height);
        stopwatch.Stop();

        Table table = new Table()
            .Title(title)
            .AddColumn("Width")
            .AddColumn("Height")
            .AddColumn("Type")
            .AddColumn("Bytes")
            .AddColumn("Elapsed");

        table.AddRow(
            mat.Width.ToString(),
            mat.Height.ToString(),
            mat.Type().ToString(),
            (mat.Step() * mat.Rows).ToString(),
            $"{stopwatch.Elapsed.TotalMilliseconds:F3} ms");

        AnsiConsole.Write(table);
    }

    private static void RunCloneBatchPreview(string title, Mat sourceMat, int batch)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        List<Mat> mats = PositiveAlgorithmHelpers.CloneMatBatch(sourceMat, batch);
        stopwatch.Stop();

        try
        {
            Table table = new Table()
                .Title(title)
                .AddColumn("Count")
                .AddColumn("Same size")
                .AddColumn("Elapsed");

            bool sameSize = mats.All(mat => mat.Width == sourceMat.Width && mat.Height == sourceMat.Height);
            table.AddRow(mats.Count.ToString(), sameSize ? "[green]yes[/]" : "[red]no[/]", $"{stopwatch.Elapsed.TotalMilliseconds:F3} ms");
            AnsiConsole.Write(table);
        }
        finally
        {
            foreach(Mat mat in mats)
                mat.Dispose();
        }
    }

    private static void PreviewFP32(
        string title,
        Mat sourceMat,
        int batch,
        InputDataToSpanBufConverter<List<Mat>, float> converter)
    {
        PreviewBuffer(title, sourceMat, batch, converter, static span => string.Join(", ", span[..Math.Min(span.Length, PreviewCount)].ToArray().Select(value => value.ToString("F4"))));
    }

    private static void PreviewFP16(
        string title,
        Mat sourceMat,
        int batch,
        InputDataToSpanBufConverter<List<Mat>, Float16> converter)
    {
        PreviewBuffer(title, sourceMat, batch, converter, static span => string.Join(", ", span[..Math.Min(span.Length, PreviewCount)].ToArray().Select(value => ((float)value).ToString("F4"))));
    }

    private static void PreviewBuffer<TBuffer>(
        string title,
        Mat sourceMat,
        int batch,
        InputDataToSpanBufConverter<List<Mat>, TBuffer> converter,
        Func<Span<TBuffer>, string> previewFormatter)
        where TBuffer : unmanaged
    {
        List<Mat> mats = PositiveAlgorithmHelpers.CloneMatBatch(sourceMat, batch);
        try
        {
            int sizeOne = sourceMat.Width * sourceMat.Height * 3;
            TBuffer[] buffer = GC.AllocateUninitializedArray<TBuffer>(sizeOne * batch);

            Stopwatch stopwatch = Stopwatch.StartNew();
            converter(mats, buffer, batch);
            stopwatch.Stop();

            Table table = new Table()
                .Title(title)
                .AddColumn("Elements")
                .AddColumn("Bytes")
                .AddColumn("Elapsed")
                .AddColumn($"First {PreviewCount}");

            table.AddRow(
                buffer.Length.ToString(),
                (buffer.Length * Unsafe.SizeOf<TBuffer>()).ToString(),
                $"{stopwatch.Elapsed.TotalMilliseconds:F3} ms",
                previewFormatter(buffer.AsSpan()));

            AnsiConsole.Write(table);
        }
        finally
        {
            foreach(Mat mat in mats)
                mat.Dispose();
        }
    }

    private static void PrintMatInfo(Mat mat)
    {
        Table table = new Table()
            .Title("Input Mat")
            .AddColumn("Width")
            .AddColumn("Height")
            .AddColumn("Channels")
            .AddColumn("Type")
            .AddColumn("Step")
            .AddColumn("Bytes");

        table.AddRow(
            mat.Width.ToString(),
            mat.Height.ToString(),
            mat.Channels().ToString(),
            mat.Type().ToString(),
            mat.Step().ToString(),
            (mat.Step() * mat.Rows).ToString());

        AnsiConsole.Write(table);
    }

    private static int GetArg(string[] args, int index, int defaultValue)
    {
        if(index >= args.Length) return defaultValue;
        return int.TryParse(args[index], out int value) && value > 0 ? value : defaultValue;
    }
}
