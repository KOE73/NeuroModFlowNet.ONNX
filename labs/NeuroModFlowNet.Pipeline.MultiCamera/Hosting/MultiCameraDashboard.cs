using OpenCvSharp;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Console dashboard (one row per camera) plus optional OpenCV preview windows. Both are driven from a single
/// loop so OpenCV's window message pump stays on one thread; Esc or Q in a preview window stops the lab.
///
/// RU: Консольный дашборд (строка на камеру) плюс опциональные окна превью OpenCV. Оба обслуживаются одним циклом,
/// чтобы message pump OpenCV оставался на одном потоке; Esc или Q в окне превью останавливает лабораторию.
/// </summary>
internal sealed class MultiCameraDashboard
{
    static readonly TimeSpan PreviewTick = TimeSpan.FromMilliseconds(40);
    static readonly TimeSpan TableTick = TimeSpan.FromMilliseconds(250);

    static readonly TimeSpan PlainTick = TimeSpan.FromSeconds(2);
    static readonly TimeSpan ProfileTick = TimeSpan.FromSeconds(30);

    readonly MultiCameraHost host;
    readonly bool showPreview;
    readonly bool plainOutput;
    readonly bool profile;

    public MultiCameraDashboard(MultiCameraHost host, bool showPreview, bool plainOutput, bool profile)
    {
        this.host = host ?? throw new ArgumentNullException(nameof(host));
        this.showPreview = showPreview;
        this.plainOutput = plainOutput;
        this.profile = profile;
    }

    public async Task RunAsync(CancellationTokenSource stopTokenSource)
    {
        ArgumentNullException.ThrowIfNull(stopTokenSource);
        CancellationToken cancellationToken = stopTokenSource.Token;

        if(plainOutput)
        {
            await RunPlainAsync(stopTokenSource).ConfigureAwait(false);
            return;
        }

        await AnsiConsole
            .Live(Render())
            .AutoClear(false)
            .StartAsync(async liveContext =>
            {
                Task hostTask = host.RunAsync(cancellationToken);
                Task uiTask = UiLoopAsync(liveContext, stopTokenSource, hostTask);

                try
                {
                    await hostTask.ConfigureAwait(false);
                }
                finally
                {
                    await uiTask.ConfigureAwait(false);
                    liveContext.UpdateTarget(Render());
                }
            })
            .ConfigureAwait(false);
    }

    /// <summary>Line-oriented output for redirected consoles and logs: one status line per camera every few seconds.</summary>
    async Task RunPlainAsync(CancellationTokenSource stopTokenSource)
    {
        Task hostTask = host.RunAsync(stopTokenSource.Token);
        DateTime nextReport = DateTime.UtcNow;
        DateTime nextProfile = DateTime.UtcNow + ProfileTick;

        while(!hostTask.IsCompleted)
        {
            if(showPreview && ShowPreviews())
                stopTokenSource.Cancel();

            if(DateTime.UtcNow >= nextReport)
            {
                foreach(CameraWorker worker in host.Workers)
                {
                    CameraStatisticsSnapshot snapshot = worker.Statistics.Snapshot();
                    Console.WriteLine(
                        $"{DateTime.Now:HH:mm:ss} {worker.Id,-12} {snapshot.Phase,-16} read {snapshot.FramesRead,6} runs {snapshot.RunsCompleted,6} " +
                        $"rej {snapshot.RunsRejected,4} fail {snapshot.RunsFailed,3} fps {snapshot.AverageFps,5:F1} ms {snapshot.AverageRunMilliseconds,6:F1} max {worker.PotentialFps(snapshot),5:F0} " +
                        $"det {snapshot.LastDetectionCount,3} trk {snapshot.LastTrackCount,3}({snapshot.LastConfirmedTrackCount}) maxId {snapshot.MaxTrackId,3} " +
                        $"ocr [{Truncate(snapshot.LastText, 40)}] {(snapshot.LastError is null ? string.Empty : "error: " + Truncate(snapshot.LastError, 120))}");
                }

                nextReport = DateTime.UtcNow + PlainTick;
            }

            if(profile && DateTime.UtcNow >= nextProfile)
            {
                PrintProfile();
                nextProfile = DateTime.UtcNow + ProfileTick;
            }

            await Task.Delay(PreviewTick).ConfigureAwait(false);
        }

        if(showPreview)
            Cv2.DestroyAllWindows();

        if(profile)
            PrintProfile();

        await hostTask.ConfigureAwait(false);
    }

    /// <summary>Per-instruction average time (EMA) per camera, slowest first, printed at exit or on demand.</summary>
    public void PrintProfile()
    {
        foreach(CameraWorker worker in host.Workers)
        {
            Console.WriteLine($"--- {worker.Id}: instruction time, EMA ms (sum = one run) ---");
            double total = 0;
            foreach((string name, double milliseconds) in worker.Statistics.SnapshotInstructionTimings())
            {
                total += milliseconds;
                Console.WriteLine($"{milliseconds,8:F2}  {name}");
            }

            Console.WriteLine($"{total,8:F2}  total");
        }
    }

    async Task UiLoopAsync(LiveDisplayContext liveContext, CancellationTokenSource stopTokenSource, Task hostTask)
    {
        DateTime nextTableRefresh = DateTime.UtcNow;

        while(!hostTask.IsCompleted)
        {
            if(showPreview && ShowPreviews())
            {
                stopTokenSource.Cancel();
                break;
            }

            if(DateTime.UtcNow >= nextTableRefresh)
            {
                liveContext.UpdateTarget(Render());
                nextTableRefresh = DateTime.UtcNow + TableTick;
            }

            await Task.Delay(PreviewTick).ConfigureAwait(false);
        }

        if(showPreview)
            Cv2.DestroyAllWindows();
    }

    /// <summary>Returns true when the user asked to stop from a preview window.</summary>
    bool ShowPreviews()
    {
        bool anyWindow = false;
        foreach(CameraWorker worker in host.Workers)
        {
            using Mat? preview = worker.TakePreviewClone();
            if(preview is null)
                continue;

            Cv2.ImShow($"{worker.Id} - rect preview", preview);
            anyWindow = true;
        }

        if(!anyWindow)
            return false;

        int key = Cv2.WaitKey(1);
        return key is 27 or 'q' or 'Q';
    }

    IRenderable Render()
    {
        var table = new Table().Border(TableBorder.Rounded).Expand();
        table.AddColumn("Camera");
        table.AddColumn("Phase");
        table.AddColumn(new TableColumn("Read").RightAligned());
        table.AddColumn(new TableColumn("Runs").RightAligned());
        table.AddColumn(new TableColumn("Rejected").RightAligned());
        table.AddColumn(new TableColumn("Failed").RightAligned());
        table.AddColumn(new TableColumn("FPS").RightAligned());
        table.AddColumn(new TableColumn("Run ms").RightAligned());
        table.AddColumn(new TableColumn("Max FPS").RightAligned());
        table.AddColumn(new TableColumn("Det").RightAligned());
        table.AddColumn(new TableColumn("Tracks (conf)").RightAligned());
        table.AddColumn(new TableColumn("Max id").RightAligned());
        table.AddColumn("Last OCR");
        table.AddColumn("Last error");

        foreach(CameraWorker worker in host.Workers)
        {
            CameraStatisticsSnapshot snapshot = worker.Statistics.Snapshot();
            table.AddRow(
                Markup.Escape(worker.Id),
                Markup.Escape(snapshot.Phase),
                snapshot.FramesRead.ToString(),
                snapshot.RunsCompleted.ToString(),
                snapshot.RunsRejected.ToString(),
                snapshot.RunsFailed.ToString(),
                snapshot.AverageFps.ToString("F1"),
                snapshot.AverageRunMilliseconds.ToString("F1"),
                worker.PotentialFps(snapshot).ToString("F0"),
                snapshot.LastDetectionCount.ToString(),
                $"{snapshot.LastTrackCount} ({snapshot.LastConfirmedTrackCount})",
                snapshot.MaxTrackId.ToString(),
                Markup.Escape(Truncate(snapshot.LastText, 40)),
                Markup.Escape(Truncate(snapshot.LastError ?? string.Empty, 60)));
        }

        return new Panel(table)
            .Header("[bold]Multi-camera VM lab[/]  FPS = actual, Run ms = one VM run incl. queueing, Max FPS = maxInFlight x 1000 / Run ms  (Esc / Q or Ctrl+C to stop)")
            .Expand();
    }

    static string Truncate(string value, int maxLength) =>
        value.Length <= maxLength ? value : string.Concat(value.AsSpan(0, maxLength - 1), "…");
}
