using FFmpeg.AutoGen.Abstractions;

namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: A retained packet in decode order. Its bytes remain valid while the packet is held.
///
/// RU: Удерживаемый пакет в порядке декодирования. Байты действительны, пока пакет удерживается.
/// </summary>
public sealed unsafe class EncodedVideoPacket : IDisposable
{
    AVPacket* packet;

    EncodedVideoPacket(AVPacket* packet, long sequence, bool randomAccess)
    {
        this.packet = packet;
        Sequence = sequence;
        IsRandomAccess = randomAccess;
        PresentationTimestamp = packet->pts;
        DecodeTimestamp = packet->dts;
        Duration = packet->duration;
        Size = packet->size;
    }

    public long Sequence { get; }
    public long PresentationTimestamp { get; }
    public long DecodeTimestamp { get; }
    public long Duration { get; }
    public int Size { get; }
    public bool IsRandomAccess { get; }
    public ReadOnlySpan<byte> Bytes => new(GetPacket()->data, Size);

    internal static EncodedVideoPacket Retain(AVPacket* packet, long sequence, bool randomAccess)
    {
        AVPacket* copy = ffmpeg.av_packet_clone(packet);
        if(copy is null)
            throw new OutOfMemoryException("Could not retain the encoded packet.");
        return new(copy, sequence, randomAccess);
    }

    /// <summary>
    /// EN: Retains an independent reference to the encoded bytes.
    ///
    /// RU: Удерживает независимую ссылку на сжатые байты.
    /// </summary>
    public EncodedVideoPacket Retain() => Retain(GetPacket(), Sequence, IsRandomAccess);

    internal AVPacket* GetPacket()
    {
        ObjectDisposedException.ThrowIf(packet is null, this);
        return packet;
    }

    public void Dispose()
    {
        AVPacket* owned = packet;
        packet = null;
        if(owned is not null)
            ffmpeg.av_packet_free(&owned);
    }
}
