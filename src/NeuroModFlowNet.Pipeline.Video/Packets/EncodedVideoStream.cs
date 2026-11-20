namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Parameters that apply to every packet produced by one encoder generation.
///
/// RU: Параметры, общие для всех пакетов одного поколения кодировщика.
/// </summary>
public sealed record EncodedVideoStream(
    string StreamId,
    Guid Generation,
    string Codec,
    string BitstreamFormat,
    int Width,
    int Height,
    int FramesPerSecond,
    int TimeBaseNumerator,
    int TimeBaseDenominator,
    ReadOnlyMemory<byte> CodecConfiguration);
