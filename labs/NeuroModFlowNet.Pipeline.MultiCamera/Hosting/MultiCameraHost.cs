namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Creates the shared endpoints and one <see cref="CameraWorker"/> per enabled camera, runs them concurrently and
/// tears everything down in the right order: controllers (and their GPU intermediates) before the endpoints.
///
/// RU: Создаёт общие endpoint-ы и по одному <see cref="CameraWorker"/> на включённую камеру, запускает их параллельно
/// и останавливает в правильном порядке: контроллеры (и их GPU intermediates) раньше endpoint-ов.
/// </summary>
internal sealed class MultiCameraHost : IAsyncDisposable
{
    readonly SharedInferenceEndpoints endpoints;
    readonly List<CameraWorker> workers = [];

    MultiCameraHost(SharedInferenceEndpoints endpoints)
    {
        this.endpoints = endpoints;
    }

    public IReadOnlyList<CameraWorker> Workers => workers;

    public static async Task<MultiCameraHost> CreateAsync(MultiCameraConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        CameraConfig[] cameras = config.EnabledCameras.ToArray();
        SharedInferenceEndpoints endpoints = SharedInferenceEndpoints.Create(config, cameras);
        var host = new MultiCameraHost(endpoints);

        try
        {
            foreach(CameraConfig camera in cameras)
                host.workers.Add(new CameraWorker(camera, config, endpoints));
        }
        catch
        {
            await host.DisposeAsync().ConfigureAwait(false);
            throw;
        }

        return host;
    }

    /// <summary>
    /// Runs every worker until cancellation. A worker failure (bad config, warmup error) cancels the others so the
    /// process fails loudly instead of silently running with fewer cameras: no fallback.
    /// </summary>
    public async Task RunAsync(CancellationToken cancellationToken)
    {
        using var linkedTokenSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var tasks = new List<Task>(workers.Count);
        Exception? firstFailure = null;

        foreach(CameraWorker worker in workers)
        {
            tasks.Add(RunWorkerAsync(worker));
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);

        if(firstFailure is not null && !cancellationToken.IsCancellationRequested)
            throw new AggregateException("One or more camera workers failed.", firstFailure);

        async Task RunWorkerAsync(CameraWorker worker)
        {
            try
            {
                await worker.RunAsync(linkedTokenSource.Token).ConfigureAwait(false);
            }
            catch(OperationCanceledException) when(linkedTokenSource.IsCancellationRequested)
            {
                // Expected on shutdown.
            }
            catch(Exception exception)
            {
                worker.Statistics.MarkRunFailed(exception.GetBaseException().Message);
                worker.Statistics.SetPhase("failed");
                Interlocked.CompareExchange(ref firstFailure, exception, null);
                linkedTokenSource.Cancel();
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        foreach(CameraWorker worker in workers)
            await worker.DisposeAsync().ConfigureAwait(false);

        workers.Clear();
        await endpoints.DisposeAsync().ConfigureAwait(false);
    }
}
