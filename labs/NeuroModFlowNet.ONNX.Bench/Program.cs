using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Bench;

internal class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        Summary summary = BenchmarkRunner.Run<Benchmark>();
        MakeTable(summary);

        // TODO Restore after legacy detector benchmarks are migrated to current runners/converters.
        //Summary detSummary = BenchmarkRunner.Run<BenchmarkDet>();
        //MakeTable(detSummary);
    }

    private static void MakeTable(Summary summary)
    {
        var fpsResults = summary.Reports
            .Where(r => r.ResultStatistics != null)
            .Select(r =>
            {
                string backend = r.BenchmarkCase.Parameters.Items.Any(p => p.Name == "_InferenceBackend")
                    ? r.BenchmarkCase.Parameters["_InferenceBackend"].ToString()
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
}
