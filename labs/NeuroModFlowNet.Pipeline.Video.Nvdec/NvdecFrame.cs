using FFmpeg.AutoGen.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: One decoded NVDEC frame kept in CUDA memory. Owns a reference to the FFmpeg hardware frame (the surface returns to
/// the decoder pool on <see cref="Dispose"/>) and exposes the NV12 planes as one zero-copy <see cref="OrtValue"/>
/// <c>[allocHeight * 3 / 2, pitch]</c> U8 over device memory. The OrtValue does not own the memory: this frame must live
/// until every VM instruction that reads the tensor has finished, so both are put into the run with
/// <c>disposeWithContext</c>.
///
/// RU: Один декодированный NVDEC-кадр в памяти CUDA. Держит ссылку на аппаратный кадр FFmpeg (поверхность возвращается в
/// пул декодера в <see cref="Dispose"/>) и отдаёт NV12-плоскости как один zero-copy <see cref="OrtValue"/>
/// <c>[allocHeight * 3 / 2, pitch]</c> U8 поверх памяти устройства. OrtValue памятью не владеет: кадр должен жить, пока
/// все инструкции VM, читающие тензор, не завершились, поэтому оба кладутся в запуск с <c>disposeWithContext</c>.
/// </summary>
public sealed unsafe class NvdecFrame : IDisposable
{
    static readonly OrtMemoryInfo CudaMemoryInfo = new(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

    AVFrame* frame;

    public NvdecFrame(AVFrame* frame, long frameIndex, TimeSpan timestamp)
    {
        this.frame = frame;
        FrameIndex = frameIndex;
        Timestamp = timestamp;

        if((AVPixelFormat)frame->format != AVPixelFormat.AV_PIX_FMT_CUDA)
            throw new InvalidOperationException($"Expected a CUDA hardware frame, got {(AVPixelFormat)frame->format}.");

        var framesContext = (AVHWFramesContext*)frame->hw_frames_ctx->data;
        if(framesContext->sw_format != AVPixelFormat.AV_PIX_FMT_NV12)
            throw new NotSupportedException($"Only 8-bit NV12 NVDEC surfaces are supported, got {framesContext->sw_format}.");

        Width = frame->width;
        Height = frame->height;
        Pitch = frame->linesize[0];
        AllocatedHeight = framesContext->height;

        // The CUDA frame pool allocates Y and UV planes back to back; the tensor view relies on that layout.
        long planeOffset = (long)frame->data[1] - (long)frame->data[0];
        if(frame->linesize[1] != Pitch || planeOffset != (long)Pitch * AllocatedHeight)
            throw new NotSupportedException($"NV12 planes are not contiguous (pitch {Pitch}/{frame->linesize[1]}, UV offset {planeOffset}, alloc height {AllocatedHeight}).");
    }

    public long FrameIndex { get; }

    public TimeSpan Timestamp { get; }

    public int Width { get; }

    public int Height { get; }

    public int Pitch { get; }

    public int AllocatedHeight { get; }

    public OrtValue CreateNv12Tensor()
    {
        ObjectDisposedException.ThrowIf(frame is null, this);

        long rows = AllocatedHeight + (AllocatedHeight / 2);
        return OrtValue.CreateTensorValueWithData(
            CudaMemoryInfo,
            TensorElementType.UInt8,
            [rows, Pitch],
            (IntPtr)frame->data[0],
            rows * Pitch);
    }

    public void Dispose()
    {
        if(frame is null)
            return;

        AVFrame* owned = frame;
        ffmpeg.av_frame_free(&owned);
        frame = null;
    }
}
