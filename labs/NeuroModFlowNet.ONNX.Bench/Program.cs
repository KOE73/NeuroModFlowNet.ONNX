using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using NeuroModFlowNet.ONNX;
using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Bench;

internal class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        InitializeOnnxNativePaths();

        if(args.Contains("--legacy-rec", StringComparer.OrdinalIgnoreCase))
        {
            Summary summary = BenchmarkRunner.Run<Benchmark>();
            MakeTable(summary);
        }
        else if(args.Contains("--fixed-capacity", StringComparer.OrdinalIgnoreCase))
        {
            Summary summary = BenchmarkRunner.Run<PaddleRecRoiPrepareFixedCapacityBenchmark>();
            MakeOcrRoiPrepareFixedCapacityTable(summary);
        }
        else if(args.Contains("--graph-variants", StringComparer.OrdinalIgnoreCase))
        {
            Summary summary = BenchmarkRunner.Run<PaddleRecRoiPrepareBenchmark>();
            MakeOcrRoiPrepareTable(summary);
        }
        else
        {
            Summary graphVariantsSummary = BenchmarkRunner.Run<PaddleRecRoiPrepareBenchmark>();
            MakeOcrRoiPrepareTable(graphVariantsSummary);

            Summary fixedCapacitySummary = BenchmarkRunner.Run<PaddleRecRoiPrepareFixedCapacityBenchmark>();
            MakeOcrRoiPrepareFixedCapacityTable(fixedCapacitySummary);
        }

        // TODO Restore after legacy detector benchmarks are migrated to current runners/converters.
        //Summary detSummary = BenchmarkRunner.Run<BenchmarkDet>();
        //MakeTable(detSummary);
    }

    private static void InitializeOnnxNativePaths()
    {
        OnnxRuntimePathHelper.InitFromConfig();

        AddNativePathFromEnvironment("NMFN_CUDA_BIN_PATH");
        AddNativePathFromEnvironment("NMFN_CUDNN_BIN_PATH");
        AddNativePathFromEnvironment("NMFN_TRT_LIB_PATH");
    }

    private static void AddNativePathFromEnvironment(string environmentVariable)
    {
        string? configuredPath = Environment.GetEnvironmentVariable(environmentVariable);
        if(string.IsNullOrWhiteSpace(configuredPath))
            return;

        string expandedPath = Environment.ExpandEnvironmentVariables(configuredPath);
        string resolvedPath = Path.IsPathRooted(expandedPath)
            ? expandedPath
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, expandedPath));

        OnnxRuntimePathHelper.AddToSystemPath(resolvedPath);
    }

    private static void MakeTable(Summary summary)
    {
        var fpsResults = summary.Reports
            .Where(r => r.ResultStatistics != null)
            .Select(r =>
            {
                string backend = r.BenchmarkCase.Parameters.Items.Any(p => p.Name == "_InferenceBackend")
                    ? r.BenchmarkCase.Parameters["_InferenceBackend"].ToString() ?? "Default"
                    : "Default";

                int batch = r.BenchmarkCase.Parameters.Items.Any(p => p.Name == "Batch")
                    ? (int)r.BenchmarkCase.Parameters["Batch"]
                    : 1;

                bool isFixed = r.BenchmarkCase.Parameters.Items.Any(p => p.Name == "FixedModel") &&
                    (bool)r.BenchmarkCase.Parameters["FixedModel"];

                double opsPerSec = 1_000_000_000.0 / r.ResultStatistics!.Mean;
                double fps = opsPerSec * batch;

                return new { Backend = backend, Batch = batch, IsFixed = isFixed, Fps = fps };
            })
            .ToList();

        var grouped = fpsResults
            .GroupBy(x => new { x.Batch, x.IsFixed })
            .OrderBy(g => g.Key.Batch)
            .ThenBy(g => g.Key.IsFixed);

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[yellow]Сводный отчет производительности (FPS)[/]")
            .Caption("[grey]FPS = (1.0 / Mean) * Batch[/]");

        table.AddColumn(new TableColumn("[u]Batch[/]").Centered());
        table.AddColumn(new TableColumn("[u]Fixed[/]").Centered());
        table.AddColumn(new TableColumn("[blue]Cuda FPS[/]").RightAligned());
        table.AddColumn(new TableColumn("[green]TensorRt FPS[/]").RightAligned());
        table.AddColumn(new TableColumn("[grey]Default FPS[/]").RightAligned());

        foreach(var group in grouped)
        {
            var cuda = group.FirstOrDefault(x => x.Backend == "Cuda")?.Fps;
            var tensorRt = group.FirstOrDefault(x => x.Backend == "TensorRt")?.Fps;
            var fallback = group.FirstOrDefault(x => x.Backend == "Default")?.Fps;

            table.AddRow(
                group.Key.Batch.ToString(),
                group.Key.IsFixed ? "[green]Fix[/]" : "[red]Dyn[/]",
                cuda.HasValue ? $"{cuda.Value:F0}" : "[grey]N/A[/]",
                tensorRt.HasValue ? $"[bold]{tensorRt.Value:F0}[/]" : "[grey]N/A[/]",
                fallback.HasValue ? $"{fallback.Value:F0}" : "[grey]N/A[/]"
            );
        }

        AnsiConsole.Write(table);
    }

    private static void MakeOcrRoiPrepareTable(Summary summary)
    {
        var rows = summary.Reports
            .Where(static r => r.ResultStatistics != null)
            .Select(static r =>
            {
                string backend = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "_InferenceBackend")
                    ? r.BenchmarkCase.Parameters["_InferenceBackend"].ToString() ?? "Unknown"
                    : "Unknown";

                string algorithm = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "Algorithm")
                    ? r.BenchmarkCase.Parameters["Algorithm"].ToString() ?? "Unknown"
                    : "Unknown";

                int regionCount = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "RegionCount")
                    ? (int)r.BenchmarkCase.Parameters["RegionCount"]
                    : 0;

                int targetWidth = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "TargetWidth")
                    ? (int)r.BenchmarkCase.Parameters["TargetWidth"]
                    : 0;

                double microseconds = r.ResultStatistics!.Mean / 1_000.0;
                double batchesPerSecond = 1_000_000.0 / microseconds;
                double roiPerSecond = batchesPerSecond * regionCount;

                return new
                {
                    Backend = backend,
                    Algorithm = algorithm,
                    RegionCount = regionCount,
                    TargetWidth = targetWidth,
                    Microseconds = microseconds,
                    RoiPerSecond = roiPerSecond
                };
            })
            .OrderBy(static x => x.Backend)
            .ThenBy(static x => x.RegionCount)
            .ThenBy(static x => x.TargetWidth)
            .ThenBy(static x => x.Algorithm)
            .ToList();

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[yellow]OCR ROI prepare ONNX graph variants[/]")
            .Caption("[grey]Input: RGB FP32 NCHW 0..1 on GPU; output: Paddle Rec FP32 NCHW batch on GPU[/]");

        table.AddColumn("[u]Backend[/]");
        table.AddColumn("[u]Algorithm[/]");
        table.AddColumn(new TableColumn("[u]ROI[/]").RightAligned());
        table.AddColumn(new TableColumn("[u]W[/]").RightAligned());
        table.AddColumn(new TableColumn("[blue]Mean us[/]").RightAligned());
        table.AddColumn(new TableColumn("[green]ROI/s[/]").RightAligned());

        foreach(var row in rows)
        {
            table.AddRow(
                row.Backend,
                row.Algorithm,
                row.RegionCount.ToString(),
                row.TargetWidth.ToString(),
                $"{row.Microseconds:F2}",
                $"{row.RoiPerSecond:F0}");
        }

        if(rows.Count > 0)
            AnsiConsole.Write(table);
    }

    private static void MakeOcrRoiPrepareFixedCapacityTable(Summary summary)
    {
        var rows = summary.Reports
            .Where(static r => r.ResultStatistics != null)
            .Select(static r =>
            {
                string backend = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "_InferenceBackend")
                    ? r.BenchmarkCase.Parameters["_InferenceBackend"].ToString() ?? "Unknown"
                    : "Unknown";

                int actualRegionCount = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "ActualRegionCount")
                    ? (int)r.BenchmarkCase.Parameters["ActualRegionCount"]
                    : 0;

                int maxRoiCount = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "MaxRoiCount")
                    ? (int)r.BenchmarkCase.Parameters["MaxRoiCount"]
                    : 0;

                int targetWidth = r.BenchmarkCase.Parameters.Items.Any(static p => p.Name == "TargetWidth")
                    ? (int)r.BenchmarkCase.Parameters["TargetWidth"]
                    : 0;

                double microseconds = r.ResultStatistics!.Mean / 1_000.0;
                double batchesPerSecond = 1_000_000.0 / microseconds;
                int effectiveRegionCount = Math.Min(actualRegionCount, maxRoiCount);
                double roiPerSecond = batchesPerSecond * effectiveRegionCount;

                return new
                {
                    Backend = backend,
                    ActualRegionCount = actualRegionCount,
                    MaxRoiCount = maxRoiCount,
                    TargetWidth = targetWidth,
                    Microseconds = microseconds,
                    RoiPerSecond = roiPerSecond
                };
            })
            .OrderBy(static x => x.Backend)
            .ThenBy(static x => x.MaxRoiCount)
            .ThenBy(static x => x.ActualRegionCount)
            .ThenBy(static x => x.TargetWidth)
            .ToList();

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[yellow]OCR ROI prepare fixed-capacity batch[/]")
            .Caption("[grey]Graph/output shape is fixed by Max ROI; Actual ROI is payload count, overflow is truncated in this benchmark[/]");

        table.AddColumn("[u]Backend[/]");
        table.AddColumn(new TableColumn("[u]Actual[/]").RightAligned());
        table.AddColumn(new TableColumn("[u]Max[/]").RightAligned());
        table.AddColumn(new TableColumn("[u]W[/]").RightAligned());
        table.AddColumn(new TableColumn("[blue]Mean us[/]").RightAligned());
        table.AddColumn(new TableColumn("[green]ROI/s[/]").RightAligned());

        foreach(var row in rows)
        {
            table.AddRow(
                row.Backend,
                row.ActualRegionCount.ToString(),
                row.MaxRoiCount.ToString(),
                row.TargetWidth.ToString(),
                $"{row.Microseconds:F2}",
                $"{row.RoiPerSecond:F0}");
        }

        if(rows.Count > 0)
            AnsiConsole.Write(table);
    }
}
