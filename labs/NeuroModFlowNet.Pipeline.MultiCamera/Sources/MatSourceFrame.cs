using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>CPU-decoded frame: registers <see cref="MultiCameraKeys.SourceImage"/> (BGR <c>Mat</c>).</summary>
internal sealed class MatSourceFrame(Mat image) : ISourceFrame
{
    public int Width => image.Width;

    public int Height => image.Height;

    public void AddTo(VmRunInputs inputs)
    {
        // The Mat is wrapped zero-copy by the VM; the frame owner releases it with the run context.
        inputs.Add(MultiCameraKeys.SourceImage, image, disposeWithContext: false);
        inputs.Add(MultiCameraKeys.SourceFrame, this, disposeWithContext: true);
    }

    public void Dispose() => image.Dispose();
}
