using Spectre.Console;
using Spectre.Console.Rendering;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Spectre.Console live dashboard for the multi-stream VM lab.
/// Displays one statistics panel per VM, a global aggregate row, and the OBB batch service panel.
/// </summary>
internal sealed class MultiStreamDashboard : IMultiStreamObserver
{
    MultiStreamHost? host;
    LiveDisplayContext? liveContext;

    int cameraIndex;
    int cameraWidth;
    int cameraHeight;
    double cameraFps;
    bool allWarmedUp;

    public async Task RunAsync(MultiStreamHost multiStreamHost, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(multiStreamHost);
        host = multiStreamHost;

        // Periodically refresh the dashboard from a timer task since the host does not call back per-frame.
        using var refreshTimer = new PeriodicTimer(TimeSpan.FromMilliseconds(200));

        await AnsiConsole
            .Live(Render())
            .AutoClear(false)
            .StartAsync(async context =>
            {
                liveContext = context;

                // Run the host and the refresh timer concurrently.
                Task hostTask = multiStreamHost.RunAsync(this, cancellationToken);
                Task refreshTask = RefreshLoopAsync(refreshTimer, cancellationToken);

                await hostTask.ConfigureAwait(false);
                await refreshTask.ConfigureAwait(false);
            })
            .ConfigureAwait(false);
    }

    async Task RefreshLoopAsync(PeriodicTimer timer, CancellationToken cancellationToken)
    {
        try
        {
            while(await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
                Refresh();
        }
        catch(OperationCanceledException)
        {
            // Expected on shutdown.
        }
    }

    public void CameraOpened(int index, int width, int height, double fps)
    {
        cameraIndex = index;
        cameraWidth = width;
        cameraHeight = height;
        cameraFps = fps;
        Refresh();
    }

    public void AllWorkersWarmedUp()
    {
        allWarmedUp = true;
        Refresh();
    }

    void Refresh()
    {
        if(liveContext is null)
            return;

        liveContext.UpdateTarget(Render());
        liveContext.Refresh();
    }

    #region Rendering

    IRenderable Render()
    {
        var rows = new Rows(
            CreateHeaderPanel(),
            CreateVmGrid(),
            CreateObbServicePanel());

        return new Panel(rows)
            .Header("[bold]Pipeline VM Camera Lab — Multi Stream[/]")
            .Border(BoxBorder.Rounded)
            .Expand();
    }

    IRenderable CreateHeaderPanel()
    {
        string status = allWarmedUp ? "[green]Running[/]" : "[yellow]Warming up…[/]";
        string cameraInfo = $"Camera {cameraIndex} ({cameraWidth}x{cameraHeight}, {cameraFps:F1} fps)";

        var table = new Table()
            .Border(TableBorder.None)
            .HideHeaders()
            .AddColumn("")
            .AddColumn("");

        table.AddRow($"[bold]Camera[/]  {cameraInfo}", $"[bold]Status[/]  {status}");
        return table;
    }

    IRenderable CreateVmGrid()
    {
        IReadOnlyList<VmWorker> workers = host?.Workers ?? [];

        if(workers.Count == 0)
            return new Markup("[grey]Initializing workers…[/]");

        var grid = new Grid();

        for(int columnIndex = 0; columnIndex < workers.Count; columnIndex++)
            grid.AddColumn(new GridColumn().PadRight(1));

        IRenderable[] vmPanels = workers
            .Select(worker => (IRenderable)CreateVmPanel(worker.Statistics))
            .ToArray();

        grid.AddRow(vmPanels);

        // Aggregate row below the per-VM panels.
        double totalFps = workers.Sum(worker => worker.Statistics.ProcessingFps);
        double avgRunMs = workers.Average(worker => worker.Statistics.AverageRunTotalMilliseconds);
        int totalFrames = workers.Sum(worker => worker.Statistics.CompletedFrames);

        var aggregateTable = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Aggregate (all VMs)[/]")
            .AddColumn("Metric")
            .AddColumn("Value");

        aggregateTable.AddRow("Total proc FPS", $"{totalFps:F1}");
        aggregateTable.AddRow("Avg run ms (mean)", $"{avgRunMs:F2}");
        aggregateTable.AddRow("Total frames", totalFrames.ToString());

        return new Rows(grid, aggregateTable);
    }

    static Panel CreateVmPanel(VmSlotStatistics statistics)
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .HideHeaders()
            .AddColumn("")
            .AddColumn("");

        table.AddRow("Frames", statistics.FailedFrames == 0
            ? statistics.CompletedFrames.ToString()
            : $"{statistics.CompletedFrames} ok / {statistics.FailedFrames} fail");
        table.AddRow("FPS", $"{statistics.ProcessingFps:F1}");
        table.AddRow("Run ms", $"{statistics.LastRunTotalMilliseconds:F1} / {statistics.AverageRunTotalMilliseconds:F1}");
        table.AddRow("Ctrl ms", $"{statistics.LastControllerTotalMilliseconds:F1} / {statistics.AverageControllerTotalMilliseconds:F1}");
        table.AddRow("Instr ms", $"{statistics.LastInstructionTotalMilliseconds:F1} / {statistics.AverageInstructionTotalMilliseconds:F1}");
        table.AddRow("Uptime", statistics.Uptime.ToString(@"hh\:mm\:ss"));

        // Instruction breakdown.
        foreach(InstructionTimingSnapshot snapshot in statistics.GetInstructionSnapshots())
            table.AddRow(Markup.Escape($"  {snapshot.Name}"), $"{snapshot.AverageMilliseconds:F2} avg");

        return new Panel(table).Header($"[bold]VM {statistics.VmIndex}[/]").Border(BoxBorder.Rounded);
    }

    IRenderable CreateObbServicePanel()
    {
        ObbBatchServiceStats stats = host?.ObbService?.GetStats() ?? ObbBatchServiceStats.Empty;

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]OBB Batch Service[/]")
            .AddColumn("Metric")
            .AddColumn("Value");

        table.AddRow("Batches processed", stats.BatchesProcessed.ToString());
        table.AddRow("Total detections", stats.TotalDetections.ToString());
        table.AddRow("Batch time ms", $"{stats.LastBatchTimeMilliseconds:F2} / {stats.AverageBatchTimeMilliseconds:F2} avg");
        table.AddRow("Detections / batch", $"{stats.LastDetectionCount:F0} / {stats.AverageDetectionCount:F1} avg");

        string batchFps = stats.AverageBatchTimeMilliseconds > 0
            ? $"{1000.0 / stats.AverageBatchTimeMilliseconds:F1}"
            : "-";
        table.AddRow("Batch FPS", batchFps);

        return table;
    }

    #endregion
}
