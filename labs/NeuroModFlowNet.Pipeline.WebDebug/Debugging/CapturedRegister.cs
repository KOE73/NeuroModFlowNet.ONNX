using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

// Owns only CPU data; no VM value, context, provider allocator or native GPU pointer may escape in this object.
internal sealed class CapturedRegister(Mat? image, DebugInspectionResult? value) : IDisposable
{
    public DebugInspectionResult Encode()
    {
        if(value is not null) return value;
        // Geometry is already bounded/prepared by the VM task. Encoding must not silently resize the owned snapshot again.
        Cv2.ImEncode(".png", image!, out byte[] encoded);
        return new("image/png", encoded);
    }
    public void Dispose() => image?.Dispose();
}
