using System.Runtime.InteropServices;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: The single explicit CUDA call of the video boundary. FFmpeg NVDEC copies a decoded surface into its frame pool
/// with <c>cuMemcpy2DAsync</c> on the legacy default stream of the device primary context. ONNX Runtime runs its CUDA /
/// TensorRT work on its own non-blocking stream, which does not implicitly wait for the legacy stream. Before a decoded
/// frame is handed to the VM, the decoder thread therefore waits for the legacy stream of that context. This only waits
/// for decode work already queued; ORT work is not blocked. <c>--no-fence</c> disables it to demonstrate the race.
///
/// RU: Единственный явный вызов CUDA на границе видео. FFmpeg NVDEC копирует декодированную поверхность в пул кадров
/// через <c>cuMemcpy2DAsync</c> в legacy default stream primary context устройства. ONNX Runtime выполняет CUDA /
/// TensorRT работу в своём non-blocking stream, который неявно не ждёт legacy stream. Поэтому перед передачей кадра в
/// VM поток декодера ждёт legacy stream этого контекста. Ждётся только уже поставленная работа декода, работа ORT не
/// блокируется. <c>--no-fence</c> выключает ожидание, чтобы показать гонку.
/// </summary>
public static class CudaDecodeFence
{
    const int CudaSuccess = 0;
    static readonly object SyncRoot = new();
    static IntPtr primaryContext;

    /// <remarks>Call after FFmpeg created its CUDA device: retaining the primary context first would activate it with
    /// default flags, and FFmpeg then refuses it ("Primary context already active with incompatible flags").</remarks>
    public static void Initialize(int deviceOrdinal)
    {
        lock(SyncRoot)
        {
            if(primaryContext != IntPtr.Zero)
                return;

            Check(cuInit(0), "cuInit");
            Check(cuDeviceGet(out int device, deviceOrdinal), "cuDeviceGet");
            Check(cuDevicePrimaryCtxRetain(out primaryContext, device), "cuDevicePrimaryCtxRetain");
        }
    }

    /// <summary>Waits for the legacy default stream of the device primary context (FFmpeg NVDEC output copies).</summary>
    public static void WaitForDecodedFrame()
    {
        if(primaryContext == IntPtr.Zero)
            throw new InvalidOperationException("CudaDecodeFence is not initialized.");

        Check(cuCtxPushCurrent_v2(primaryContext), "cuCtxPushCurrent");
        try
        {
            Check(cuStreamSynchronize(IntPtr.Zero), "cuStreamSynchronize");
        }
        finally
        {
            cuCtxPopCurrent_v2(out _);
        }
    }

    static void Check(int result, string call)
    {
        if(result != CudaSuccess)
            throw new InvalidOperationException($"CUDA driver call {call} failed with code {result}.");
    }

    [DllImport("nvcuda.dll")] static extern int cuInit(uint flags);

    [DllImport("nvcuda.dll")] static extern int cuDeviceGet(out int device, int ordinal);

    [DllImport("nvcuda.dll")] static extern int cuDevicePrimaryCtxRetain(out IntPtr context, int device);

    [DllImport("nvcuda.dll")] static extern int cuCtxPushCurrent_v2(IntPtr context);

    [DllImport("nvcuda.dll")] static extern int cuCtxPopCurrent_v2(out IntPtr context);

    [DllImport("nvcuda.dll")] static extern int cuStreamSynchronize(IntPtr stream);
}
