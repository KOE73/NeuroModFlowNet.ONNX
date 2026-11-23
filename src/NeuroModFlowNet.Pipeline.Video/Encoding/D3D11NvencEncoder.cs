using System.Runtime.Versioning;
using FFmpeg.AutoGen.Abstractions;
using NeuroModFlowNet.Video.FFmpeg;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed unsafe class D3D11NvencEncoder : IDisposable
{
    const int ClockRate = 90_000;
    readonly VideoStreamProfile profile;
    AVCodecContext* codec;
    AVPacket* packet;
    long sequence;
    bool disposed;

    public D3D11NvencEncoder(VideoStreamProfile profile, NvencSurface firstSurface)
    {
        this.profile = profile;
        try
        {
            Open(firstSurface.Frame);
            if(codec->extradata is null || codec->extradata_size <= 0)
                throw new InvalidOperationException("NVENC did not provide global H.264 headers.");
            Description = new(profile.StreamId, Guid.NewGuid(), "h264", "annex-b",
                profile.OutputWidth, profile.OutputHeight, profile.FramesPerSecond, 1, ClockRate,
                new ReadOnlySpan<byte>(codec->extradata, codec->extradata_size).ToArray());
        }
        catch { Dispose(); throw; }
    }

    public EncodedVideoStream Description { get; }

    #region Encoder initialization

    void Open(AVFrame* firstSurface)
    {
        if(firstSurface is null || firstSurface->format != (int)AVPixelFormat.AV_PIX_FMT_D3D11 || firstSurface->hw_frames_ctx is null)
            throw new InvalidOperationException("NVENC requires a prepared D3D11 NV12 surface.");
        var frames = (AVHWFramesContext*)firstSurface->hw_frames_ctx->data;
        if(frames->sw_format != AVPixelFormat.AV_PIX_FMT_NV12 || frames->width != profile.OutputWidth || frames->height != profile.OutputHeight)
            throw new InvalidDataException("The prepared surface does not match the output profile.");

        AVCodec* encoder = ffmpeg.avcodec_find_encoder_by_name("h264_nvenc");
        if(encoder is null)
            throw new NotSupportedException("FFmpeg h264_nvenc is unavailable.");
        codec = ffmpeg.avcodec_alloc_context3(encoder);
        packet = ffmpeg.av_packet_alloc();
        if(codec is null || packet is null)
            throw new OutOfMemoryException("Could not allocate H.264 encoder state.");
        codec->width = profile.OutputWidth;
        codec->height = profile.OutputHeight;
        codec->pix_fmt = AVPixelFormat.AV_PIX_FMT_D3D11;
        codec->hw_frames_ctx = ffmpeg.av_buffer_ref(firstSurface->hw_frames_ctx);
        if(codec->hw_frames_ctx is null)
            throw new OutOfMemoryException("Could not retain the NVENC surface pool.");
        codec->time_base = new() { num = 1, den = ClockRate };
        codec->framerate = new() { num = profile.FramesPerSecond, den = 1 };
        codec->sample_aspect_ratio = new() { num = 1, den = 1 };
        codec->color_range = AVColorRange.AVCOL_RANGE_MPEG;
        codec->colorspace = AVColorSpace.AVCOL_SPC_BT709;
        codec->color_primaries = AVColorPrimaries.AVCOL_PRI_BT709;
        codec->color_trc = AVColorTransferCharacteristic.AVCOL_TRC_BT709;
        codec->chroma_sample_location = AVChromaLocation.AVCHROMA_LOC_LEFT;
        codec->gop_size = profile.FramesPerSecond;
        codec->max_b_frames = 0;
        codec->bit_rate = profile.Bitrate;
        codec->flags |= ffmpeg.AV_CODEC_FLAG_GLOBAL_HEADER;
        FfmpegRuntime.Check(ffmpeg.av_opt_set(codec->priv_data, "preset", "p1", 0), "set NVENC preset");
        FfmpegRuntime.Check(ffmpeg.av_opt_set(codec->priv_data, "tune", "ull", 0), "set NVENC tuning");
        FfmpegRuntime.Check(ffmpeg.av_opt_set_int(codec->priv_data, "delay", 0, 0), "set NVENC delay");
        FfmpegRuntime.Check(ffmpeg.av_opt_set_int(codec->priv_data, "zerolatency", 1, 0), "set NVENC low latency");
        FfmpegRuntime.Check(ffmpeg.avcodec_open2(codec, encoder, null), "open H.264 NVENC");
    }

    #endregion

    #region Packets and shutdown

    public void Encode(NvencSurface surface, long tick, Action<EncodedVideoPacket> publish)
    {
        AVFrame* frame = surface.Frame;
        if(frame->hw_frames_ctx is null || frame->hw_frames_ctx->data != codec->hw_frames_ctx->data)
            throw new InvalidDataException("The prepared surface belongs to a different hardware pool.");
        frame->pts = checked(tick * ClockRate) / profile.FramesPerSecond;
        frame->duration = checked((tick + 1) * ClockRate) / profile.FramesPerSecond - frame->pts;
        FfmpegRuntime.Check(ffmpeg.avcodec_send_frame(codec, frame), "submit NVENC surface");
        Receive(publish);
    }

    public void Flush(Action<EncodedVideoPacket> publish)
    {
        int result = ffmpeg.avcodec_send_frame(codec, null);
        if(result != ffmpeg.AVERROR_EOF)
        {
            FfmpegRuntime.Check(result, "flush NVENC");
            Receive(publish);
        }
    }

    void Receive(Action<EncodedVideoPacket> publish)
    {
        while(true)
        {
            int result = ffmpeg.avcodec_receive_packet(codec, packet);
            if(result == FfmpegRuntime.ErrorAgain || result == ffmpeg.AVERROR_EOF)
                return;
            FfmpegRuntime.Check(result, "receive NVENC packet");
            try
            {
                bool randomAccess = (packet->flags & ffmpeg.AV_PKT_FLAG_KEY) != 0 && ContainsIdr(new(packet->data, packet->size));
                using EncodedVideoPacket output = EncodedVideoPacket.Retain(packet, sequence++, randomAccess);
                publish(output);
            }
            finally { ffmpeg.av_packet_unref(packet); }
        }
    }

    static bool ContainsIdr(ReadOnlySpan<byte> bytes)
    {
        for(int index = 0; index + 3 < bytes.Length; index++)
        {
            if(bytes[index] == 0 && bytes[index + 1] == 0 && bytes[index + 2] == 1 &&
                (bytes[index + 3] & 31) == 5)
                return true;
        }
        return false;
    }

    public void Dispose()
    {
        if(disposed)
            return;
        disposed = true;
        AVPacket* ownedPacket = packet;
        ffmpeg.av_packet_free(&ownedPacket);
        AVCodecContext* ownedCodec = codec;
        ffmpeg.avcodec_free_context(&ownedCodec);
    }

    #endregion
}
