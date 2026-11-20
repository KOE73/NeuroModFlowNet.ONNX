namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: An independently scheduled packet destination and its queue limits.
///
/// RU: Независимый получатель пакетов и ограничения его очереди.
/// </summary>
public sealed record VideoPacketOutput(
    string Id,
    Func<IVideoPacketSink> CreateSink,
    int MaximumQueueBytes = 8 * 1024 * 1024,
    int MaximumQueueMilliseconds = 2000,
    int MaximumQueuePackets = 256,
    bool Reconnect = false);
