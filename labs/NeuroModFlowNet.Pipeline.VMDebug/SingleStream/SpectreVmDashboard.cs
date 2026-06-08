using Spectre.Console;
using Spectre.Console.Rendering;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Spectre.Console live dashboard for the single-stream camera VM lab.
/// </summary>
internal sealed class SpectreVmDashboard : IVmCameraObserver
{
    readonly VmLiveStatistics statistics = new();
    LiveDisplayContext? liveContext;

    public async Task RunAsync(CameraVmHost host, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(host);

        await AnsiConsole
            .Live(Render())
            .AutoClear(false)
            .StartAsync(async context =>
            {
                liveContext = context;
                await host.RunAsync(this, cancellationToken).ConfigureAwait(false);
            })
            .ConfigureAwait(false);
    }

    public void CameraOpened(int cameraIndex, int width, int height, double fps)
    {
        statistics.SetCamera(cameraIndex, width, height, fps);
        Refresh();
    }

    public void WarmupCompleted(VmRunOutcome outcome)
    {
        statistics.AddWarmup(outcome);
        Refresh();
    }

    public void FrameCompleted(int frameIndex, VmRunOutcome outcome, TimeSpan runTotalTime)
    {
        statistics.AddFrame(outcome, runTotalTime);
        Refresh();
    }

    void Refresh()
    {
        if(liveContext is null)
            return;

        liveContext.UpdateTarget(Render());
        liveContext.Refresh();
    }

    IRenderable Render()
    {
        var grid = new Grid();
        grid.AddColumn(new GridColumn().PadRight(2));
        grid.AddColumn();
        grid.AddRow(CreateSummaryTable(), CreateInstructionTable());

        return new Panel(grid)
            .Header("[bold]Pipeline VM Camera Lab — Single Stream[/]")
            .Border(BoxBorder.Rounded)
            .Expand();
    }

    Table CreateSummaryTable()
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]VM[/]")
            .AddColumn("Metric")
            .AddColumn("Value");

        table.AddRow("Camera", $"{statistics.CameraIndex} ({statistics.CameraWidth}x{statistics.CameraHeight}, {statistics.CameraFps:F1} fps)");
        table.AddRow("Frames", statistics.FailedFrames == 0
            ? statistics.CompletedFrames.ToString()
            : $"{statistics.CompletedFrames} ok / {statistics.FailedFrames} failed");
        table.AddRow("Proc FPS", $"{statistics.ProcessingFps:F1}");
        table.AddRow("Run total ms", $"{statistics.LastRunTotalMilliseconds:F2} / {statistics.AverageRunTotalMilliseconds:F2} avg");
        table.AddRow("VM trace ms", $"{statistics.LastInstructionTotalMilliseconds:F2} / {statistics.AverageInstructionTotalMilliseconds:F2} avg");
        table.AddRow("Outside trace ms", $"{statistics.LastOutsideTraceMilliseconds:F2} / {statistics.AverageOutsideTraceMilliseconds:F2} avg");
        table.AddRow("Controller total", $"{statistics.LastControllerTotalMilliseconds:F2} / {statistics.AverageControllerTotalMilliseconds:F2} avg");
        table.AddRow("Program", $"{statistics.LastProgramMilliseconds:F2} / {statistics.AverageProgramMilliseconds:F2} avg");
        table.AddRow("Capture output", $"{statistics.LastOutputCaptureMilliseconds:F2} / {statistics.AverageOutputCaptureMilliseconds:F2} avg");
        table.AddRow("Dispose context", $"{statistics.LastContextDisposeMilliseconds:F2} / {statistics.AverageContextDisposeMilliseconds:F2} avg");
        table.AddRow("Task gap", $"{statistics.LastTaskScheduleGapMilliseconds:F2} / {statistics.AverageTaskScheduleGapMilliseconds:F2} avg");
        table.AddRow("Uptime", statistics.Uptime.ToString(@"hh\:mm\:ss"));

        return table;
    }

    Table CreateInstructionTable()
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Instruction timings, moving avg 60[/]")
            .AddColumn("Instruction")
            .AddColumn("Last ms")
            .AddColumn("Avg ms");

        foreach(InstructionTimingSnapshot snapshot in statistics.InstructionSnapshots)
        {
            table.AddRow(
                Markup.Escape(snapshot.Name),
                snapshot.LastMilliseconds.ToString("F3"),
                snapshot.AverageMilliseconds.ToString("F3"));
        }

        if(statistics.InstructionSnapshots.Count == 0)
            table.AddRow("[grey]waiting[/]", "-", "-");

        return table;
    }
}
