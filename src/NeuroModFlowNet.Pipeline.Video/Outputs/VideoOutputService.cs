using System.Runtime.Versioning;
using NeuroModFlowNet.Video.FFmpeg;

namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Schedules optional GPU video output independently from the producer. Post consumes the frame even if it is rejected.
/// Drain the service before disposing the allocator(s) that produced its tensors.
///
/// RU: Планирует необязательный GPU-видеовыход независимо от источника. Post принимает владение и при пропуске кадра.
/// Завершите службу до освобождения аллокаторов, создавших входные тензоры.
/// </summary>
[SupportedOSPlatform("windows")]
public sealed class VideoOutputService : IAsyncDisposable
{
    readonly Dictionary<string, VideoStreamWorker> streams;
    readonly SemaphoreSlim gpuWork;
    int disposed;

    public VideoOutputService(string ffmpegLibrariesPath, string rendererLibraryPath,
        IEnumerable<VideoStreamProfile> profiles, int maximumConcurrentGpuWork = 1, Action<VideoOutputEvent>? notify = null)
        : this(ffmpegLibrariesPath, rendererLibraryPath, profiles, maximumConcurrentGpuWork, notify, resolveDrawer: null)
    {
    }

    /// <param name="resolveDrawer">
    /// EN: Host resolver called once for each render VM whose profile specifies a drawer key.
    /// RU: Функция хоста, вызываемая один раз для каждой render VM, профиль которой задаёт ключ рисовальщика.
    /// </param>
    public VideoOutputService(string ffmpegLibrariesPath, string rendererLibraryPath,
        IEnumerable<VideoStreamProfile> profiles, Func<string, IDirect2DDrawer> resolveDrawer,
        int maximumConcurrentGpuWork = 1, Action<VideoOutputEvent>? notify = null)
        : this(ffmpegLibrariesPath, rendererLibraryPath, profiles, maximumConcurrentGpuWork, notify,
            resolveDrawer ?? throw new ArgumentNullException(nameof(resolveDrawer)))
    {
    }

    VideoOutputService(string ffmpegLibrariesPath, string rendererLibraryPath,
        IEnumerable<VideoStreamProfile> profiles, int maximumConcurrentGpuWork, Action<VideoOutputEvent>? notify,
        Func<string, IDirect2DDrawer>? resolveDrawer)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumConcurrentGpuWork);
        ArgumentNullException.ThrowIfNull(profiles);
        var configurations = profiles.Select(static profile => profile with { Outputs = profile.Outputs.ToArray() })
            .ToDictionary(static profile => profile.StreamId, StringComparer.Ordinal);
        foreach(VideoStreamProfile profile in configurations.Values)
            Validate(profile);
        string renderer = Path.GetFullPath(rendererLibraryPath);
        if(!File.Exists(renderer))
            throw new FileNotFoundException("The Direct2D renderer library is required.", renderer);

        // Resolve all keys before starting any workers. The bound objects live for the lifetime of their render VMs.
        var drawers = configurations.Values.ToDictionary(static profile => profile.StreamId,
            profile => BindDrawer(profile, resolveDrawer), StringComparer.Ordinal);
        FfmpegRuntime.Initialize(ffmpegLibrariesPath);
        gpuWork = new(maximumConcurrentGpuWork, maximumConcurrentGpuWork);
        streams = configurations.Values.ToDictionary(static profile => profile.StreamId,
            profile => new VideoStreamWorker(profile, renderer, drawers[profile.StreamId], gpuWork, notify), StringComparer.Ordinal);
    }

    public bool IsAccepting(string streamId) => Volatile.Read(ref disposed) == 0 &&
        streams.TryGetValue(streamId, out VideoStreamWorker? stream) && stream.IsAccepting;

    public bool Post(string streamId, VideoInputFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if(streamId is null || Volatile.Read(ref disposed) != 0 || !streams.TryGetValue(streamId, out VideoStreamWorker? stream))
        {
            frame.Dispose();
            return false;
        }
        return stream.Post(frame);
    }

    public async ValueTask DisposeAsync()
    {
        if(Interlocked.Exchange(ref disposed, 1) != 0)
            return;
        try { await Task.WhenAll(streams.Values.Select(stream => stream.DisposeAsync().AsTask())).ConfigureAwait(false); }
        finally { gpuWork.Dispose(); }
    }

    static void Validate(VideoStreamProfile profile)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(profile.StreamId);
        if(profile.InputWidth is <= 0 or > 8192 || profile.InputHeight is <= 0 or > 8192 ||
            profile.OutputWidth is <= 0 or > 8192 || profile.OutputHeight is <= 0 or > 8192 ||
            (profile.OutputWidth & 1) != 0 || (profile.OutputHeight & 1) != 0 ||
            profile.FramesPerSecond is <= 0 or > 240 || profile.CudaDeviceId < 0 || profile.Bitrate <= 0 ||
            profile.MaximumFrameAgeMilliseconds <= 0 || profile.Outputs.Count == 0)
            throw new ArgumentException($"Invalid video profile '{profile.StreamId}'.");

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach(VideoPacketOutput output in profile.Outputs)
        {
            if(string.IsNullOrWhiteSpace(output.Id) || !ids.Add(output.Id) || output.CreateSink is null ||
                output.MaximumQueueBytes <= 0 || output.MaximumQueueMilliseconds <= 0 || output.MaximumQueuePackets <= 0)
                throw new ArgumentException($"Invalid packet output in '{profile.StreamId}'.");
        }
    }

    static IDirect2DDrawer BindDrawer(VideoStreamProfile profile, Func<string, IDirect2DDrawer>? resolveDrawer)
    {
        if(profile.DrawerName is not string name)
            return new RectangleVideoDrawer();
        ArgumentException.ThrowIfNullOrWhiteSpace(name, nameof(profile.DrawerName));
        if(resolveDrawer is null)
            throw new InvalidOperationException($"The render VM for '{profile.StreamId}' requires a resolver for drawer '{name}'.");

        try
        {
            return resolveDrawer(name) ?? throw new InvalidOperationException("The host resolver returned no drawer.");
        }
        catch(InvalidOperationException exception)
        {
            throw new InvalidOperationException($"Could not bind Direct2D drawer '{name}' for '{profile.StreamId}'.", exception);
        }
    }
}
