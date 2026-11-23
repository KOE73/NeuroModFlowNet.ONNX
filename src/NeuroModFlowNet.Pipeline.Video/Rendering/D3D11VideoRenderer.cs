using System.Runtime.Versioning;
using FFmpeg.AutoGen.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed unsafe class D3D11VideoRenderer : IDisposable
{
    readonly Direct2DFrameRenderer renderer;
    readonly OrtMemoryInfo memory;
    readonly OnnxExecutionContext packing;
    readonly int deviceId;
    readonly long[] inputShape;
    readonly long[] packedShape;
    readonly long packedSize;
    bool pending;
    bool disposed;

    public D3D11VideoRenderer(VideoStreamProfile profile, string rendererLibraryPath)
    {
        deviceId = profile.CudaDeviceId;
        inputShape = [1, profile.InputHeight, profile.InputWidth, 3];
        packedShape = [1, profile.InputHeight, profile.InputWidth, 4];
        packedSize = (long)profile.InputHeight * profile.InputWidth * 4;
        memory = new(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
        try
        {
            renderer = new(rendererLibraryPath, deviceId, profile.InputWidth, profile.InputHeight,
                profile.OutputWidth, profile.OutputHeight, requireStagedDrawing: true, requireCanvasDrawing: true);
            var key = new RuntimeOnnxOperatorKernelKey("BgrU8Nhwc_To_BgraU8Nhwc",
                $"{profile.InputWidth}x{profile.InputHeight}", InferenceBackend.Cuda, $"device_id={deviceId}");
            packing = RuntimeOnnxOperatorKernelCache.CreateContext(key,
                () => BgrToBgraBuilder.Build(profile.InputWidth, profile.InputHeight),
                config => config.DeviceId = deviceId, "BGR to BGRA");
        }
        catch { Dispose(); throw; }
    }

    public nint Device => renderer.Device;

    public void Prepare(OrtValue input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if(pending)
            throw new InvalidOperationException("Complete or reset the current video frame before preparing another.");

        var info = input.GetTensorTypeAndShape();
        using OrtMemoryInfo placement = input.GetTensorMemoryInfo();
        if(info.ElementDataType != TensorElementType.UInt8 || !info.Shape.AsSpan().SequenceEqual(inputShape) ||
            placement.Name != "Cuda" || placement.Id != deviceId)
            throw new InvalidDataException("The video profile requires BGR U8 NHWC in its declared CUDA device and shape.");

        pending = true;
        nint mapped = renderer.MapPackedBgra();
        using(OrtValue destination = OrtValue.CreateTensorValueWithData(memory, TensorElementType.UInt8, packedShape, mapped, packedSize))
        {
            try
            {
                packing.IoBinding.BindInput("bgr", input);
                packing.IoBinding.BindOutput("bgra", destination);
                packing.Model.Session.RunWithBinding(packing.RunOptions, packing.IoBinding);
                packing.IoBinding.SynchronizeBoundOutputs();
            }
            finally
            {
                packing.IoBinding.ClearBoundInputs();
                packing.IoBinding.ClearBoundOutputs();
            }
        }
        renderer.PrepareBackground();
    }

    public IDirect2DCanvas BeginDraw() => renderer.BeginDraw();

    public void EndDraw(NvencSurface target)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        AVFrame* frame = target.Frame;
        if(frame is null || frame->format != (int)AVPixelFormat.AV_PIX_FMT_D3D11 || frame->data[0] is null)
            throw new InvalidOperationException("The video target must be a live D3D11 encoder surface.");

        renderer.EndDraw((nint)frame->data[0], checked((uint)(nuint)frame->data[1]));
        pending = false;
    }

    public void Reset()
    {
        if(!pending)
            return;

        renderer.Reset();
        pending = false;
    }

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;
        pending = false;
        packing?.Dispose();
        renderer?.Dispose();
        memory.Dispose();
    }
}
