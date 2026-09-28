using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: NVDEC frame: registers <see cref="MultiCameraKeys.SourceNv12"/>, a zero-copy NV12 tensor over the CUDA surface.
/// The tensor view and the surface are released together when the run context disposes this frame.
///
/// RU: Кадр NVDEC: регистр <see cref="MultiCameraKeys.SourceNv12"/>, zero-copy NV12-тензор поверх поверхности CUDA.
/// Тензор и поверхность освобождаются вместе, когда контекст запуска освобождает этот кадр.
/// </summary>
internal sealed class NvdecSourceFrame : ISourceFrame
{
    readonly NvdecFrame frame;
    readonly OrtValue nv12;

    public NvdecSourceFrame(NvdecFrame frame)
    {
        this.frame = frame;
        nv12 = frame.CreateNv12Tensor();
    }

    public int Width => frame.Width;

    public int Height => frame.Height;

    public void AddTo(VmRunInputs inputs)
    {
        inputs.Add(MultiCameraKeys.SourceNv12, nv12, disposeWithContext: false);
        inputs.Add(MultiCameraKeys.SourceFrame, this, disposeWithContext: true);
    }

    public void Dispose()
    {
        nv12.Dispose();
        frame.Dispose();
    }
}
