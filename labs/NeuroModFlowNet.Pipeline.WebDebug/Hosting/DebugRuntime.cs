using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.MultiCamera;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Owns a separate debug run of the existing multi-camera host, with one in-flight frame per camera.
///
/// RU: Владеет отдельным отладочным запуском существующего многокамерного хоста, с одним кадром в работе на камеру.
/// </summary>
internal sealed class DebugRuntime(IConfiguration configuration, ILogger<DebugRuntime> logger) : IAsyncDisposable
{
    readonly object syncRoot = new();
    readonly SemaphoreSlim lifecycle = new(1, 1);
    public BatchDiagnostics Batches { get; } = new();
    readonly Dictionary<string, (CameraWorker Worker, CameraDebugSession Session, CameraConfig Config)> cameras = new(StringComparer.Ordinal);
    CancellationTokenSource? stop;
    Task? execution;
    string state = "stopped";
    string? error;
    string? backend;

    #region Public API

    public object Snapshot()
    {
        lock(syncRoot)
            return new
            {
                name = "NeuroModFlowNet VM Debug", state, error, backend,
                canStart = execution is null || execution.IsCompleted,
                canStop = execution is { IsCompleted: false },
                cameras = cameras.Select(pair => new
                {
                    id = pair.Key, decoder = pair.Value.Config.Source.Decoder,
                    sourceKind = pair.Value.Config.Source.Kind,
                    width = pair.Value.Config.Resolution.Width, height = pair.Value.Config.Resolution.Height,
                    maxInFlight = 1, statistics = pair.Value.Worker.Statistics.Snapshot(), debug = pair.Value.Session.Snapshot()
                }).ToArray()
            };
    }

    public CameraDebugSession Session(string id)
    {
        lock(syncRoot)
            return cameras.TryGetValue(id, out var camera) ? camera.Session : throw new KeyNotFoundException("Camera not found.");
    }

    public byte[] Preview(string id)
    {
        lock(syncRoot)
        {
            if(!cameras.TryGetValue(id, out var camera)) throw new KeyNotFoundException("Camera not found.");
            using var preview = camera.Worker.TakePreviewClone();
            return preview is null ? throw new KeyNotFoundException("No completed frame preview yet.") : TensorImagePreviewRenderer.EncodePng(preview);
        }
    }

    public async Task StartAsync()
    {
        await lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            if(execution is { IsCompleted: false }) throw new InvalidOperationException("The debugger is already starting or running.");
            if(execution is not null) await execution.ConfigureAwait(false);
            stop?.Dispose();
            stop = new CancellationTokenSource();
            lock(syncRoot) { state = "starting"; error = null; }
            execution = Task.Run(() => RunAsync(stop.Token));
        }
        finally { lifecycle.Release(); }
    }

    public async Task StopAsync()
    {
        await lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            stop?.Cancel();
            ReleasePauses();
            if(execution is not null) await execution.ConfigureAwait(false);
            lock(syncRoot) { if(state != "failed") state = "stopped"; }
        }
        finally { lifecycle.Release(); }
    }

    #endregion

    #region Host ownership

    async Task RunAsync(CancellationToken cancellationToken)
    {
        SharedInferenceEndpoints? endpoints = null;
        var workers = new List<CameraWorker>();
        var sessions = new List<CameraDebugSession>();
        try
        {
            OnnxRuntimePathHelper.InitFromConfig();
            RuntimeOnnxOperatorKernelCache.SessionSharing = RuntimeKernelSessionSharing.PerCaller;
            string configPath = configuration["WebDebug:CamerasConfig"]
                ?? throw new InvalidOperationException("Set WebDebug:CamerasConfig to an existing multi-camera JSON configuration.");
            MultiCameraConfig config = MultiCameraConfigLoader.Load(Path.GetFullPath(configPath));
            string? decoder = configuration["WebDebug:Decoder"];
            config = config with
            {
                Cameras = config.Cameras.Select(camera => camera with
                {
                    MaxInFlight = 1,
                    Source = decoder is null ? camera.Source : camera.Source with { Decoder = decoder, HwAcceleration = "none" }
                }).ToArray()
            };
            config.Validate();
            if(config.EnabledCameras.Any(camera => camera.Source.IsNvdec))
            {
                string libraries = configuration["WebDebug:FFmpegLibrariesPath"]
                    ?? System.Configuration.ConfigurationManager.AppSettings["FFmpegLibrariesPath"]
                    ?? throw new InvalidOperationException("WebDebug:FFmpegLibrariesPath is required for NVDEC.");
                FfmpegRuntime.Initialize(libraries);
            }
            CameraConfig[] enabled = config.EnabledCameras.ToArray();
            Batches.Reset();
            endpoints = SharedInferenceEndpoints.Create(config, enabled, Batches);
            lock(syncRoot) { cameras.Clear(); backend = config.Backend.ToString(); }
            foreach(CameraConfig camera in enabled)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var inspector = new PausedRegisterInspector(config.Backend, camera.Source.ColorMatrix, camera.Resolution.Width, camera.Resolution.Height);
                var session = new CameraDebugSession(camera.Id, inspector);
                sessions.Add(session);
                var worker = new CameraWorker(camera, config, endpoints, session.Decorate, outcome =>
                {
                    session.RunCompleted(outcome);
                    if(outcome.Status == VmRunStatus.Failed)
                    {
                        Fail(outcome.Exception ?? new InvalidOperationException("VM run failed."));
                        stop?.Cancel();
                        ReleasePauses();
                    }
                });
                workers.Add(worker);
                lock(syncRoot) cameras.Add(camera.Id, (worker, session, camera));
            }
            lock(syncRoot) state = "running";
            Task[] tasks = workers.Select(worker => RunWorkerAsync(worker, cancellationToken)).ToArray();
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
        catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested) { }
        catch(Exception exception) { Fail(exception); }
        finally
        {
            ReleasePauses();
            // Readers/runs have finished before controllers release their allocators; endpoints are last.
            foreach(CameraWorker worker in workers)
            {
                try { await worker.DisposeAsync().ConfigureAwait(false); }
                catch(Exception exception) { Fail(exception); }
            }
            if(endpoints is not null)
            {
                try { await endpoints.DisposeAsync().ConfigureAwait(false); }
                catch(Exception exception) { Fail(exception); }
            }
            lock(syncRoot)
            {
                cameras.Clear();
                if(state != "failed") state = "stopped";
            }
            foreach(CameraDebugSession session in sessions) session.Dispose();
        }
    }

    async Task RunWorkerAsync(CameraWorker worker, CancellationToken cancellationToken)
    {
        try
        {
            await worker.RunAsync(cancellationToken).ConfigureAwait(false);
            if(worker.Statistics.Snapshot() is { RunsFailed: > 0 } failed)
                throw new InvalidOperationException(failed.LastError ?? "Camera failed.");
        }
        catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested) { }
        catch(Exception exception)
        {
            Fail(exception);
            stop?.Cancel();
            ReleasePauses();
        }
    }

    void ReleasePauses()
    {
        lock(syncRoot)
            foreach(var camera in cameras.Values)
                camera.Session.Command(new("detach"));
    }

    void Fail(Exception exception)
    {
        logger.LogError(exception, "Debug camera run failed");
        lock(syncRoot) { state = "failed"; error = exception.GetBaseException().Message; }
    }

    #endregion

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        stop?.Dispose();
        lifecycle.Dispose();
    }
}
