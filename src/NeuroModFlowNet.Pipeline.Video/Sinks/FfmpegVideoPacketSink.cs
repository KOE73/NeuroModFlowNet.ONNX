using System.Diagnostics;
using FFmpeg.AutoGen.Abstractions;
using NeuroModFlowNet.Video.FFmpeg;

namespace NeuroModFlowNet.Pipeline.Video;

internal sealed unsafe class FfmpegVideoPacketSink : IVideoPacketSink
{
    readonly string format;
    readonly string destination;
    readonly int timeoutMilliseconds;
    readonly bool network;
    readonly AVIOInterruptCB_callback interrupt;
    AVFormatContext* mux;
    AVStream* track;
    AVRational sourceTimeBase;
    CancellationToken token;
    long deadline;
    long? firstDecodeTimestamp;
    bool headerWritten;

    public FfmpegVideoPacketSink(string format, string destination, int timeoutMilliseconds, bool network)
    {
        this.format = format;
        this.destination = destination;
        this.timeoutMilliseconds = timeoutMilliseconds;
        this.network = network;
        interrupt = Interrupt;
    }

    #region Mux lifecycle

    public void Open(EncodedVideoStream stream, CancellationToken cancellationToken)
    {
        if(stream.Codec != "h264" || stream.BitstreamFormat != "annex-b" || stream.CodecConfiguration.IsEmpty)
            throw new NotSupportedException("This output requires H.264 Annex B and codec parameters.");

        AVFormatContext* created = null;
        int result = ffmpeg.avformat_alloc_output_context2(&created, null, format, network ? destination : null);
        mux = created;
        FfmpegRuntime.Check(result, "allocate output");
        if(mux is null)
            throw new InvalidOperationException("FFmpeg did not allocate an output.");
        mux->interrupt_callback.callback = interrupt;
        track = ffmpeg.avformat_new_stream(mux, null);
        if(track is null)
            throw new OutOfMemoryException("Could not allocate a muxer track.");
        sourceTimeBase = new() { num = stream.TimeBaseNumerator, den = stream.TimeBaseDenominator };
        track->time_base = sourceTimeBase;
        track->avg_frame_rate = new() { num = stream.FramesPerSecond, den = 1 };
        AVCodecParameters* parameters = track->codecpar;
        parameters->codec_type = AVMediaType.AVMEDIA_TYPE_VIDEO;
        parameters->codec_id = AVCodecID.AV_CODEC_ID_H264;
        parameters->width = stream.Width;
        parameters->height = stream.Height;
        parameters->color_range = AVColorRange.AVCOL_RANGE_MPEG;
        parameters->color_space = AVColorSpace.AVCOL_SPC_BT709;
        parameters->color_primaries = AVColorPrimaries.AVCOL_PRI_BT709;
        parameters->color_trc = AVColorTransferCharacteristic.AVCOL_TRC_BT709;
        int size = stream.CodecConfiguration.Length;
        parameters->extradata = (byte*)ffmpeg.av_mallocz((ulong)size + ffmpeg.AV_INPUT_BUFFER_PADDING_SIZE);
        if(parameters->extradata is null)
            throw new OutOfMemoryException("Could not allocate codec parameters.");
        stream.CodecConfiguration.Span.CopyTo(new Span<byte>(parameters->extradata, size));
        parameters->extradata_size = size;

        BeginIo(cancellationToken);
        AVDictionary* options = null;
        try
        {
            if(network)
                FfmpegRuntime.Check(ffmpeg.av_dict_set(&options, "rtsp_transport", "tcp", 0), "configure RTSP transport");
            else
            {
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                using(new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) { }
                FfmpegRuntime.Check(ffmpeg.avio_open2(&mux->pb, destination, ffmpeg.AVIO_FLAG_WRITE,
                    &mux->interrupt_callback, null), "open MP4 file");
            }

            FfmpegRuntime.Check(ffmpeg.avformat_write_header(mux, &options), "write stream header");
            headerWritten = true;
        }
        finally
        {
            ffmpeg.av_dict_free(&options);
            EndIo();
        }
    }

    public void Write(EncodedVideoPacket packet, CancellationToken cancellationToken)
    {
        AVPacket* copy = ffmpeg.av_packet_clone(packet.GetPacket());
        if(copy is null)
            throw new OutOfMemoryException("Could not hold a muxer packet.");
        try
        {
            firstDecodeTimestamp ??= copy->dts != ffmpeg.AV_NOPTS_VALUE ? copy->dts : 0;
            if(copy->pts != ffmpeg.AV_NOPTS_VALUE)
                copy->pts -= firstDecodeTimestamp.Value;
            if(copy->dts != ffmpeg.AV_NOPTS_VALUE)
                copy->dts -= firstDecodeTimestamp.Value;
            ffmpeg.av_packet_rescale_ts(copy, sourceTimeBase, track->time_base);
            copy->stream_index = track->index;
            BeginIo(cancellationToken);
            FfmpegRuntime.Check(ffmpeg.av_interleaved_write_frame(mux, copy), "write encoded packet");
        }
        finally
        {
            EndIo();
            ffmpeg.av_packet_free(&copy);
        }
    }

    public void Dispose()
    {
        if(mux is null)
            return;
        BeginIo(CancellationToken.None);
        try
        {
            if(headerWritten)
                FfmpegRuntime.Check(ffmpeg.av_write_trailer(mux), "finalize output");
        }
        finally
        {
            try
            {
                if(!network && mux->pb is not null)
                    FfmpegRuntime.Check(ffmpeg.avio_closep(&mux->pb), "close MP4 file");
            }
            finally
            {
                ffmpeg.avformat_free_context(mux);
                mux = null;
                EndIo();
                GC.KeepAlive(interrupt);
            }
        }
    }

    #endregion

    void BeginIo(CancellationToken cancellationToken)
    {
        token = cancellationToken;
        Volatile.Write(ref deadline, Stopwatch.GetTimestamp() + Stopwatch.Frequency * timeoutMilliseconds / 1000L);
    }

    void EndIo() => Volatile.Write(ref deadline, 0);

    int Interrupt(void* opaque)
    {
        long activeDeadline = Volatile.Read(ref deadline);
        return activeDeadline != 0 && (token.IsCancellationRequested || Stopwatch.GetTimestamp() >= activeDeadline) ? 1 : 0;
    }
}
