namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Runtime settings for the single-stream camera lab host.
/// </summary>
internal sealed record CameraVmHostOptions(
    int CameraIndex,
    int MaxInFlight,
    bool ShowOpenCvWindows,
    bool SyncLoopToCameraFps)
{
    public static CameraVmHostOptions Default { get; } = new(
        CameraIndex: 0,
        MaxInFlight: 1,
        ShowOpenCvWindows: true,
        SyncLoopToCameraFps: true);

    public static CameraVmHostOptions FromSettings(LabSettings settings) => new(
        CameraIndex: settings.CameraIndex,
        MaxInFlight: settings.SingleStream.MaxInFlight,
        ShowOpenCvWindows: settings.ShowOpenCvWindows,
        SyncLoopToCameraFps: settings.SyncLoopToCameraFps);
}
