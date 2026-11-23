namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: One explicitly configured CUDA BGR U8 NHWC input and H.264 NVENC output stream.
///
/// RU: Один явно настроенный вход CUDA BGR U8 NHWC и выходной поток H.264 NVENC.
/// </summary>
public sealed record VideoStreamProfile(
    string StreamId,
    int InputWidth,
    int InputHeight,
    int OutputWidth,
    int OutputHeight,
    int FramesPerSecond,
    int Bitrate,
    IReadOnlyList<VideoPacketOutput> Outputs,
    int CudaDeviceId = 0,
    int MaximumFrameAgeMilliseconds = 2000,
    bool RepeatLastFrame = true)
{
    /// <summary>
    /// EN: Key of the drawer bound to this profile when the host builds its render VM.
    ///
    /// RU: Ключ рисовальщика, который хост привязывает к этому профилю при сборке render VM.
    /// </summary>
    public string? DrawerName { get; init; }
}
