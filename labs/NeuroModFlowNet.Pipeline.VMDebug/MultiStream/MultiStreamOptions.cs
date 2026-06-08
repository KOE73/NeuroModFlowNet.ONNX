namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Runtime options for the multi-stream lab host.
/// </summary>
internal sealed record MultiStreamOptions(
    int CameraIndex,
    int VmCount,
    int ObbBatchSize,
    int MaxInFlightPerVm,
    bool ShowOpenCvWindows,
    bool SyncLoopToCameraFps)
{
    /// <summary>
    /// Default: 4 VM workers, batch size 4 (one frame per VM per batch), single in-flight per VM.
    /// </summary>
    public static MultiStreamOptions Default { get; } = new(
        CameraIndex: 0,
        VmCount: 4,
        ObbBatchSize: 4,
        MaxInFlightPerVm: 1,
        ShowOpenCvWindows: true,
        SyncLoopToCameraFps: true);

    public static MultiStreamOptions FromSettings(LabSettings settings) => new(
        CameraIndex: settings.CameraIndex,
        VmCount: settings.MultiStream.VmCount,
        ObbBatchSize: settings.MultiStream.ObbBatchSize,
        MaxInFlightPerVm: settings.MultiStream.MaxInFlightPerVm,
        ShowOpenCvWindows: settings.ShowOpenCvWindows,
        SyncLoopToCameraFps: settings.SyncLoopToCameraFps);
}
