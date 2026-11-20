using FFmpeg.AutoGen.Abstractions;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>FFmpeg owns the CUDA surfaces. ORT writes NV12 into them using bound outputs; NVENC consumes them on the encoder task.</summary>
internal sealed unsafe class CudaH264Encoder : IDisposable
{
    readonly int width, height;
    readonly OrtMemoryInfo memory = new(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
    AVBufferRef* frames;
    AVCodecContext* codec;
    AVPacket* packet;
    OnnxExecutionContext? conversion;
    int pitch;
    bool initialized;
    bool retainedPrimary;

    public CudaH264Encoder(string libraries, int width, int height)
    {
        this.width = width;
        this.height = height;
        if(width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
            throw new ArgumentException("H.264 requires even image dimensions.");
        try
        {
            FfmpegRuntime.Initialize(libraries);
            initialized = true;
            Open();
            using Frame prototype = Allocate();
            pitch = prototype.Pointer->linesize[0];
            var key = new RuntimeOnnxOperatorKernelKey("DebugBgrToNv12", $"{width}x{height};pitch={pitch}", InferenceBackend.Cuda, "bt709-limited");
            conversion = RuntimeOnnxOperatorKernelCache.CreateContext(key, () => BgrToNv12Graph.Build(width, height, pitch),
                configure: null, displayName: "debug-bgr-to-nv12.onnx");
        }
        catch { Dispose(); throw; }
    }

    void Open()
    {
        AVDictionary* options = null;
        AVBufferRef* device = null;
        CheckCuda(cuInit(0));
        CheckCuda(cuDevicePrimaryCtxRetain(out nint primary, 0));
        retainedPrimary = true;
        CheckCuda(cuCtxPushCurrent_v2(primary));
        try
        {
            // Reuse ORT's active primary context without changing its scheduling flags.
            ffmpeg.av_dict_set(&options, "current_ctx", "1", 0);
            FfmpegRuntime.Check(ffmpeg.av_hwdevice_ctx_create(&device, AVHWDeviceType.AV_HWDEVICE_TYPE_CUDA, "0", options, 0), "create encoder CUDA device");
            frames = ffmpeg.av_hwframe_ctx_alloc(device);
            if(frames is null) throw new InvalidOperationException("Could not allocate CUDA frame pool.");
            var pool = (AVHWFramesContext*)frames->data;
            pool->format = AVPixelFormat.AV_PIX_FMT_CUDA;
            pool->sw_format = AVPixelFormat.AV_PIX_FMT_NV12;
            pool->width = width;
            pool->height = height;
            FfmpegRuntime.Check(ffmpeg.av_hwframe_ctx_init(frames), "init encoder CUDA frame pool");
        }
        finally { ffmpeg.av_dict_free(&options); ffmpeg.av_buffer_unref(&device); CheckCuda(cuCtxPopCurrent_v2(out _)); }

        AVCodec* encoder = ffmpeg.avcodec_find_encoder_by_name("h264_nvenc");
        if(encoder is null) throw new NotSupportedException("FFmpeg h264_nvenc is unavailable.");
        codec = ffmpeg.avcodec_alloc_context3(encoder);
        packet = ffmpeg.av_packet_alloc();
        if(codec is null || packet is null) throw new InvalidOperationException("Could not allocate encoder resources.");
        codec->width = width;
        codec->height = height;
        codec->pix_fmt = AVPixelFormat.AV_PIX_FMT_CUDA;
        codec->hw_frames_ctx = ffmpeg.av_buffer_ref(frames);
        codec->time_base = new AVRational { num = 1, den = 1000 };
        codec->framerate = new AVRational { num = 25, den = 1 };
        codec->colorspace = AVColorSpace.AVCOL_SPC_BT709;
        codec->color_range = AVColorRange.AVCOL_RANGE_MPEG;
        codec->gop_size = 25;
        codec->max_b_frames = 0;
        codec->bit_rate = 2_000_000;
        try
        {
            ffmpeg.av_dict_set(&options, "preset", "p1", 0);
            ffmpeg.av_dict_set(&options, "tune", "ull", 0);
            ffmpeg.av_dict_set(&options, "zerolatency", "1", 0);
            ffmpeg.av_dict_set(&options, "delay", "0", 0);
            ffmpeg.av_dict_set(&options, "profile", "main", 0);
            FfmpegRuntime.Check(ffmpeg.avcodec_open2(codec, encoder, &options), "open h264_nvenc");
        }
        finally { ffmpeg.av_dict_free(&options); }
    }

    Frame Allocate()
    {
        AVFrame* pointer = ffmpeg.av_frame_alloc();
        if(pointer is null) throw new InvalidOperationException("Could not allocate CUDA frame.");
        var frame = new Frame(pointer);
        try { FfmpegRuntime.Check(ffmpeg.av_hwframe_get_buffer(frames, pointer, 0), "rent CUDA frame"); return frame; }
        catch { frame.Dispose(); throw; }
    }

    public Frame Capture(OrtValue input)
    {
        var info = input.GetTensorTypeAndShape();
        using OrtMemoryInfo placement = input.GetTensorMemoryInfo();
        if(info.ElementDataType != TensorElementType.UInt8 || !info.Shape.AsSpan().SequenceEqual([1L, height, width, 3]) ||
            placement.Name != "Cuda" || placement.Id != 0)
            throw new NotSupportedException("H.264 requires a CUDA:0 tensor in BGR U8 NHWC.");

        Frame frame = Allocate();
        try
        {
            AVFrame* surface = frame.Pointer;
            if(surface->linesize[0] != pitch || surface->linesize[1] != pitch)
                throw new InvalidOperationException("Unexpected NV12 plane pitch.");
            using OrtValue y = OrtValue.CreateTensorValueWithData(memory, TensorElementType.UInt8, [height, pitch], (nint)surface->data[0], (long)height * pitch);
            using OrtValue uv = OrtValue.CreateTensorValueWithData(memory, TensorElementType.UInt8, [height / 2, pitch], (nint)surface->data[1], (long)(height / 2) * pitch);
            var context = conversion!;
            try
            {
                context.IoBinding.BindInput("bgr", input);
                context.IoBinding.BindOutput("y", y);
                context.IoBinding.BindOutput("uv", uv);
                context.Model.Session.RunWithBinding(context.RunOptions, context.IoBinding);
                context.IoBinding.SynchronizeBoundOutputs();
            }
            finally { context.IoBinding.ClearBoundInputs(); context.IoBinding.ClearBoundOutputs(); }
            return frame;
        }
        catch { frame.Dispose(); throw; }
    }

    public void Encode(Frame frame, long timestamp, Action<long, bool, byte[]> publish)
    {
        frame.Pointer->pts = timestamp;
        FfmpegRuntime.Check(ffmpeg.avcodec_send_frame(codec, frame.Pointer), "submit NVENC frame");
        while(true)
        {
            int result = ffmpeg.avcodec_receive_packet(codec, packet);
            if(result == FfmpegRuntime.ErrorAgain) return;
            FfmpegRuntime.Check(result, "receive NVENC packet");
            try { publish(packet->pts, (packet->flags & ffmpeg.AV_PKT_FLAG_KEY) != 0, new ReadOnlySpan<byte>(packet->data, packet->size).ToArray()); }
            finally { ffmpeg.av_packet_unref(packet); }
        }
    }

    public void Dispose()
    {
        conversion?.Dispose();
        if(initialized)
        {
            AVPacket* ownedPacket = packet;
            ffmpeg.av_packet_free(&ownedPacket);
            AVCodecContext* ownedCodec = codec;
            ffmpeg.avcodec_free_context(&ownedCodec);
            AVBufferRef* ownedFrames = frames;
            ffmpeg.av_buffer_unref(&ownedFrames);
        }
        if(retainedPrimary) cuDevicePrimaryCtxRelease_v2(0);
        memory.Dispose();
    }

    static void CheckCuda(int result)
    {
        if(result != 0) throw new InvalidOperationException($"CUDA context binding failed ({result}).");
    }

    [DllImport("nvcuda.dll")] static extern int cuInit(uint flags);
    [DllImport("nvcuda.dll")] static extern int cuDevicePrimaryCtxRetain(out nint context, int device);
    [DllImport("nvcuda.dll")] static extern int cuDevicePrimaryCtxRelease_v2(int device);
    [DllImport("nvcuda.dll")] static extern int cuCtxPushCurrent_v2(nint context);
    [DllImport("nvcuda.dll")] static extern int cuCtxPopCurrent_v2(out nint context);

    internal sealed class Frame(AVFrame* pointer) : IDisposable
    {
        public AVFrame* Pointer => pointer;
        public void Dispose()
        {
            AVFrame* owned = pointer;
            pointer = null;
            ffmpeg.av_frame_free(&owned);
        }
    }
}
