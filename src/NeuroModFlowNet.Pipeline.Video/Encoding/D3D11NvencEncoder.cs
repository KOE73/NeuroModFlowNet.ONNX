using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using FFmpeg.AutoGen.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Video.FFmpeg;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed unsafe class D3D11NvencEncoder : IDisposable
{
    const int ClockRate = 90_000;
    readonly VideoStreamProfile profile;
    readonly Direct2DFrameRenderer renderer;
    readonly OrtMemoryInfo memory;
    readonly long[] inputShape;
    readonly long[] packedShape;
    readonly long packedSize;
    AVBufferRef* surfaces;
    AVCodecContext* codec;
    AVPacket* packet;
    OnnxExecutionContext? packing;
    long sequence;
    bool disposed;

    public D3D11NvencEncoder(VideoStreamProfile profile, string rendererLibraryPath)
    {
        this.profile = profile;
        inputShape = [1, profile.InputHeight, profile.InputWidth, 3];
        packedShape = [1, profile.InputHeight, profile.InputWidth, 4];
        packedSize = (long)profile.InputHeight * profile.InputWidth * 4;
        memory = new(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, profile.CudaDeviceId, OrtMemType.Default);
        try
        {
            renderer = new(rendererLibraryPath, profile.CudaDeviceId,
                profile.InputWidth, profile.InputHeight, profile.OutputWidth, profile.OutputHeight);
            Open();
            var key = new RuntimeOnnxOperatorKernelKey("BgrU8Nhwc_To_BgraU8Nhwc",
                $"{profile.InputWidth}x{profile.InputHeight}", InferenceBackend.Cuda, $"device_id={profile.CudaDeviceId}");
            packing = RuntimeOnnxOperatorKernelCache.CreateContext(key,
                () => BgrToBgraBuilder.Build(profile.InputWidth, profile.InputHeight),
                config => config.DeviceId = profile.CudaDeviceId, "BGR to BGRA");
            if(codec->extradata is null || codec->extradata_size <= 0)
                throw new InvalidOperationException("NVENC did not provide global H.264 headers.");
            Description = new(profile.StreamId, Guid.NewGuid(), "h264", "annex-b",
                profile.OutputWidth, profile.OutputHeight, profile.FramesPerSecond, 1, ClockRate,
                new ReadOnlySpan<byte>(codec->extradata, codec->extradata_size).ToArray());
        }
        catch { Dispose(); throw; }
    }

    public EncodedVideoStream Description { get; }

    #region Encoder and GPU surfaces

    void Open()
    {
        AVBufferRef* device = ffmpeg.av_hwdevice_ctx_alloc(AVHWDeviceType.AV_HWDEVICE_TYPE_D3D11VA);
        if(device is null)
            throw new OutOfMemoryException("Could not allocate the FFmpeg D3D11 device context.");
        try
        {
            var hardware = (AVHWDeviceContext*)device->data;
            var d3d = (AVD3D11VADeviceContext*)hardware->hwctx;
            nint pointer = renderer.Device;
            Marshal.AddRef(pointer);
            d3d->device = (ID3D11Device*)pointer;
            FfmpegRuntime.Check(ffmpeg.av_hwdevice_ctx_init(device), "initialize shared D3D11 device");

            surfaces = ffmpeg.av_hwframe_ctx_alloc(device);
            if(surfaces is null)
                throw new OutOfMemoryException("Could not allocate the NVENC surface pool.");
            var frames = (AVHWFramesContext*)surfaces->data;
            frames->format = AVPixelFormat.AV_PIX_FMT_D3D11;
            frames->sw_format = AVPixelFormat.AV_PIX_FMT_NV12;
            frames->width = profile.OutputWidth;
            frames->height = profile.OutputHeight;
            const uint renderTargetBinding = 0x20;
            ((AVD3D11VAFramesContext*)frames->hwctx)->BindFlags = renderTargetBinding;
            FfmpegRuntime.Check(ffmpeg.av_hwframe_ctx_init(surfaces), "initialize NVENC surfaces");
        }
        finally { ffmpeg.av_buffer_unref(&device); }

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
        codec->hw_frames_ctx = ffmpeg.av_buffer_ref(surfaces);
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

    public NvencSurface Render(VideoInputFrame frame)
    {
        OrtValue input = frame.Tensor;
        var info = input.GetTensorTypeAndShape();
        using OrtMemoryInfo placement = input.GetTensorMemoryInfo();
        if(info.ElementDataType != TensorElementType.UInt8 || !info.Shape.AsSpan().SequenceEqual(inputShape) ||
            placement.Name != "Cuda" || placement.Id != profile.CudaDeviceId)
            throw new InvalidDataException("The video profile requires BGR U8 NHWC in its declared CUDA device and shape.");

        AVFrame* target = ffmpeg.av_frame_alloc();
        if(target is null)
            throw new OutOfMemoryException("Could not allocate a surface reference.");
        var surface = new NvencSurface(target);
        try
        {
            FfmpegRuntime.Check(ffmpeg.av_hwframe_get_buffer(surfaces, target, 0), "rent NV12 surface");
            if(target->format != (int)AVPixelFormat.AV_PIX_FMT_D3D11)
                throw new InvalidOperationException("The encoder surface is not D3D11.");
            nint mapped = renderer.MapPackedBgra();
            using(OrtValue destination = OrtValue.CreateTensorValueWithData(memory, TensorElementType.UInt8, packedShape, mapped, packedSize))
            {
                OnnxExecutionContext context = packing!;
                try
                {
                    context.IoBinding.BindInput("bgr", input);
                    context.IoBinding.BindOutput("bgra", destination);
                    context.Model.Session.RunWithBinding(context.RunOptions, context.IoBinding);
                    context.IoBinding.SynchronizeBoundOutputs();
                }
                finally
                {
                    context.IoBinding.ClearBoundInputs();
                    context.IoBinding.ClearBoundOutputs();
                }
            }
            renderer.Draw((nint)target->data[0], checked((uint)(nuint)target->data[1]), frame.Rectangles.Span);
            return surface;
        }
        catch { surface.Dispose(); throw; }
    }

    #endregion

    #region Packets and shutdown

    public void Encode(NvencSurface surface, long tick, Action<EncodedVideoPacket> publish)
    {
        AVFrame* frame = surface.Frame;
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
        packing?.Dispose();
        AVPacket* ownedPacket = packet;
        ffmpeg.av_packet_free(&ownedPacket);
        AVCodecContext* ownedCodec = codec;
        ffmpeg.avcodec_free_context(&ownedCodec);
        AVBufferRef* ownedSurfaces = surfaces;
        ffmpeg.av_buffer_unref(&ownedSurfaces);
        renderer?.Dispose();
        memory.Dispose();
    }

    #endregion
}
