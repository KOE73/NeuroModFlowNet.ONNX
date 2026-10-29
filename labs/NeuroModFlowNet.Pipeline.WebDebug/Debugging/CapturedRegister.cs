using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

// Owns only CPU data; no VM value, context, provider allocator or native GPU pointer may escape in this object.
internal sealed class CapturedRegister(Mat? image, DebugInspectionResult? value) : IDisposable
{
    public DebugInspectionResult Encode() => value ?? new("image/png", TensorImagePreviewRenderer.EncodePng(image!, 2000));
    public void Dispose() => image?.Dispose();
}
