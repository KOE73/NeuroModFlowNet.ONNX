using System.Runtime.InteropServices;
using FFmpeg.AutoGen.Abstractions;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: FFmpeg demux + native decoder with the NVDEC hwaccel (CUDA device on the primary context, shared with ONNX
/// Runtime). Frames stay in CUDA memory and are returned as <see cref="NvdecFrame"/>. Files can loop; RTSP uses TCP.
/// No fallback: if the decoder does not offer <c>AV_PIX_FMT_CUDA</c>, decoding fails instead of switching to CPU.
///
/// RU: Демультиплексор FFmpeg + нативный декодер с hwaccel NVDEC (CUDA-устройство на primary context, общем с ONNX
/// Runtime). Кадры остаются в памяти CUDA и возвращаются как <see cref="NvdecFrame"/>. Файлы могут зацикливаться,
/// RTSP идёт по TCP. Без fallback: если декодер не предлагает <c>AV_PIX_FMT_CUDA</c>, декод падает, а не уходит на CPU.
/// </summary>
/// <remarks>
/// EN: Not thread-safe; one instance per camera thread. <c>extraHardwareFrames</c> enlarges the surface pool so that
/// frames held by in-flight VM runs do not starve the decoder.
///
/// RU: Не потокобезопасен; один экземпляр на поток камеры. <c>extraHardwareFrames</c> увеличивает пул поверхностей,
/// чтобы кадры, удерживаемые запусками VM, не останавливали декодер.
/// </remarks>
public sealed unsafe class NvdecVideoDecoder : IDisposable
{
    // Kept alive for the process lifetime: FFmpeg stores the native function pointer.
    static readonly AVCodecContext_get_format GetFormatCallback = SelectCudaFormat;

    readonly string source;
    readonly bool loop;
    readonly bool useFence;
    AVFormatContext* formatContext;
    AVCodecContext* codecContext;
    AVBufferRef* deviceContext;
    AVPacket* packet;
    AVFrame* receiveFrame;
    int streamIndex;
    AVRational timeBase;
    bool draining;
    long frameIndex;

    public NvdecVideoDecoder(string source, bool loop, bool useFence = true, int extraHardwareFrames = 8, int deviceOrdinal = 0)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        this.source = source;
        this.loop = loop;
        this.useFence = useFence;

        try
        {
            OpenInput();
            OpenDecoder(extraHardwareFrames, deviceOrdinal);
            packet = ffmpeg.av_packet_alloc();
            receiveFrame = ffmpeg.av_frame_alloc();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public int Width => codecContext->width;

    public int Height => codecContext->height;

    public double Fps { get; private set; }

    public string CodecName { get; private set; } = string.Empty;

    #region Open

    void OpenInput()
    {
        AVDictionary* options = null;
        if(source.StartsWith("rtsp://", StringComparison.OrdinalIgnoreCase))
        {
            ffmpeg.av_dict_set(&options, "rtsp_transport", "tcp", 0);
            ffmpeg.av_dict_set(&options, "timeout", "5000000", 0);
        }

        AVFormatContext* context = null;
        try
        {
            FfmpegRuntime.Check(ffmpeg.avformat_open_input(&context, source, null, &options), $"open '{source}'");
        }
        finally
        {
            ffmpeg.av_dict_free(&options);
        }

        formatContext = context;
        FfmpegRuntime.Check(ffmpeg.avformat_find_stream_info(formatContext, null), "find stream info");

        AVCodec* codec = null;
        streamIndex = FfmpegRuntime.Check(
            ffmpeg.av_find_best_stream(formatContext, AVMediaType.AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0),
            "find video stream");

        AVStream* stream = formatContext->streams[streamIndex];
        timeBase = stream->time_base;
        AVRational rate = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
        Fps = rate.den > 0 ? rate.num / (double)rate.den : 0;
        CodecName = ffmpeg.avcodec_get_name(stream->codecpar->codec_id);
    }

    void OpenDecoder(int extraHardwareFrames, int deviceOrdinal)
    {
        AVStream* stream = formatContext->streams[streamIndex];
        AVCodec* decoder = ffmpeg.avcodec_find_decoder(stream->codecpar->codec_id);
        if(decoder is null)
            throw new NotSupportedException($"No FFmpeg decoder for codec {CodecName}.");

        if(!SupportsCuda(decoder))
            throw new NotSupportedException($"Decoder '{new string((sbyte*)decoder->name)}' has no CUDA (NVDEC) hwaccel in this FFmpeg build.");

        // primary_ctx=1: decode on the device primary context, the same one ONNX Runtime's CUDA/TensorRT EP uses, so
        // device pointers of decoded surfaces are directly valid for ORT tensors.
        AVDictionary* deviceOptions = null;
        ffmpeg.av_dict_set(&deviceOptions, "primary_ctx", "1", 0);
        AVBufferRef* device = null;
        try
        {
            FfmpegRuntime.Check(
                ffmpeg.av_hwdevice_ctx_create(&device, AVHWDeviceType.AV_HWDEVICE_TYPE_CUDA, deviceOrdinal.ToString(), deviceOptions, 0),
                "create CUDA device");
        }
        finally
        {
            ffmpeg.av_dict_free(&deviceOptions);
        }

        deviceContext = device;
        codecContext = ffmpeg.avcodec_alloc_context3(decoder);
        FfmpegRuntime.Check(ffmpeg.avcodec_parameters_to_context(codecContext, stream->codecpar), "copy codec parameters");
        codecContext->hw_device_ctx = ffmpeg.av_buffer_ref(deviceContext);
        codecContext->get_format = GetFormatCallback;
        codecContext->extra_hw_frames = extraHardwareFrames;
        codecContext->pkt_timebase = stream->time_base;
        FfmpegRuntime.Check(ffmpeg.avcodec_open2(codecContext, decoder, null), "open decoder");

        // FFmpeg must be the first to activate the primary context (it sets its scheduling flags); the fence only
        // retains the already active context afterwards.
        if(useFence)
            CudaDecodeFence.Initialize(deviceOrdinal);
    }

    static bool SupportsCuda(AVCodec* decoder)
    {
        for(int index = 0; ; index++)
        {
            AVCodecHWConfig* config = ffmpeg.avcodec_get_hw_config(decoder, index);
            if(config is null)
                return false;

            // 0x01 = AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX
            if(config->device_type == AVHWDeviceType.AV_HWDEVICE_TYPE_CUDA && (config->methods & 0x01) != 0)
                return true;
        }
    }

    static AVPixelFormat SelectCudaFormat(AVCodecContext* context, AVPixelFormat* formats)
    {
        for(AVPixelFormat* format = formats; *format != AVPixelFormat.AV_PIX_FMT_NONE; format++)
        {
            if(*format == AVPixelFormat.AV_PIX_FMT_CUDA)
                return AVPixelFormat.AV_PIX_FMT_CUDA;
        }

        // Explicitly refuse software output: returning NONE makes the decode fail instead of silently using the CPU.
        return AVPixelFormat.AV_PIX_FMT_NONE;
    }

    #endregion

    #region Decode

    /// <summary>Returns the next decoded frame, or <c>null</c> at the end of a non-looping source. Caller owns the frame.</summary>
    public NvdecFrame? ReadNext()
    {
        while(true)
        {
            int result = ffmpeg.avcodec_receive_frame(codecContext, receiveFrame);
            if(result == 0)
                return TakeFrame();

            if(result == ffmpeg.AVERROR_EOF)
            {
                if(!loop)
                    return null;

                RewindToStart();
                continue;
            }

            if(result != FfmpegRuntime.ErrorAgain)
                FfmpegRuntime.Check(result, "receive frame");

            result = ffmpeg.av_read_frame(formatContext, packet);
            if(result == ffmpeg.AVERROR_EOF)
            {
                if(!draining)
                {
                    FfmpegRuntime.Check(ffmpeg.avcodec_send_packet(codecContext, null), "flush decoder");
                    draining = true;
                }

                continue;
            }

            FfmpegRuntime.Check(result, "read packet");
            try
            {
                if(packet->stream_index == streamIndex)
                    FfmpegRuntime.Check(ffmpeg.avcodec_send_packet(codecContext, packet), "send packet");
            }
            finally
            {
                ffmpeg.av_packet_unref(packet);
            }
        }
    }

    NvdecFrame TakeFrame()
    {
        AVFrame* owned = ffmpeg.av_frame_clone(receiveFrame);
        ffmpeg.av_frame_unref(receiveFrame);
        if(owned is null)
            throw new OutOfMemoryException("av_frame_clone failed.");

        long pts = owned->best_effort_timestamp;
        TimeSpan timestamp = pts == ffmpeg.AV_NOPTS_VALUE
            ? TimeSpan.Zero
            : TimeSpan.FromSeconds(pts * ffmpeg.av_q2d(timeBase));

        if(useFence)
            CudaDecodeFence.WaitForDecodedFrame();

        try
        {
            return new NvdecFrame(owned, frameIndex++, timestamp);
        }
        catch
        {
            ffmpeg.av_frame_free(&owned);
            throw;
        }
    }

    void RewindToStart()
    {
        FfmpegRuntime.Check(ffmpeg.av_seek_frame(formatContext, streamIndex, 0, ffmpeg.AVSEEK_FLAG_BACKWARD), "seek to start");
        ffmpeg.avcodec_flush_buffers(codecContext);
        draining = false;
    }

    #endregion

    public void Dispose()
    {
        if(receiveFrame is not null)
        {
            AVFrame* frame = receiveFrame;
            ffmpeg.av_frame_free(&frame);
            receiveFrame = null;
        }

        if(packet is not null)
        {
            AVPacket* ownedPacket = packet;
            ffmpeg.av_packet_free(&ownedPacket);
            packet = null;
        }

        if(codecContext is not null)
        {
            AVCodecContext* context = codecContext;
            ffmpeg.avcodec_free_context(&context);
            codecContext = null;
        }

        if(deviceContext is not null)
        {
            AVBufferRef* device = deviceContext;
            ffmpeg.av_buffer_unref(&device);
            deviceContext = null;
        }

        if(formatContext is not null)
        {
            AVFormatContext* context = formatContext;
            ffmpeg.avformat_close_input(&context);
            formatContext = null;
        }
    }
}
