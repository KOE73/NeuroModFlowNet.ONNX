using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using FFmpeg.AutoGen.Abstractions;
using NeuroModFlowNet.Video.FFmpeg;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed unsafe class D3D11Nv12SurfacePool : IDisposable
{
    AVBufferRef* surfaces;

    public D3D11Nv12SurfacePool(nint d3d11Device, int width, int height)
    {
        AVBufferRef* device = ffmpeg.av_hwdevice_ctx_alloc(AVHWDeviceType.AV_HWDEVICE_TYPE_D3D11VA);
        if(device is null)
            throw new OutOfMemoryException("Could not allocate the FFmpeg D3D11 device context.");

        try
        {
            var hardware = (AVHWDeviceContext*)device->data;
            var d3d = (AVD3D11VADeviceContext*)hardware->hwctx;
            Marshal.AddRef(d3d11Device);
            d3d->device = (ID3D11Device*)d3d11Device;
            FfmpegRuntime.Check(ffmpeg.av_hwdevice_ctx_init(device), "initialize shared D3D11 device");

            surfaces = ffmpeg.av_hwframe_ctx_alloc(device);
            if(surfaces is null)
                throw new OutOfMemoryException("Could not allocate the D3D11 surface pool.");
            var frames = (AVHWFramesContext*)surfaces->data;
            frames->format = AVPixelFormat.AV_PIX_FMT_D3D11;
            frames->sw_format = AVPixelFormat.AV_PIX_FMT_NV12;
            frames->width = width;
            frames->height = height;
            const uint renderTargetBinding = 0x20;
            ((AVD3D11VAFramesContext*)frames->hwctx)->BindFlags = renderTargetBinding;
            FfmpegRuntime.Check(ffmpeg.av_hwframe_ctx_init(surfaces), "initialize D3D11 NV12 surfaces");
        }
        catch { Dispose(); throw; }
        finally { ffmpeg.av_buffer_unref(&device); }
    }

    public NvencSurface Rent()
    {
        ObjectDisposedException.ThrowIf(surfaces is null, this);
        AVFrame* target = ffmpeg.av_frame_alloc();
        if(target is null)
            throw new OutOfMemoryException("Could not allocate a surface reference.");

        var surface = new NvencSurface(target);
        try
        {
            FfmpegRuntime.Check(ffmpeg.av_hwframe_get_buffer(surfaces, target, 0), "rent NV12 surface");
            if(target->format != (int)AVPixelFormat.AV_PIX_FMT_D3D11)
                throw new InvalidOperationException("The encoder surface is not D3D11.");
            return surface;
        }
        catch { surface.Dispose(); throw; }
    }

    public void Dispose()
    {
        AVBufferRef* owned = surfaces;
        surfaces = null;
        ffmpeg.av_buffer_unref(&owned);
    }
}
