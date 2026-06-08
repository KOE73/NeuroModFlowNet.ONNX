using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX;

public interface IOnnxModelOutputs
{
    OrtValue GetOutputValue(string? name = null);
    ReadOnlySpan<T> GetTensorDataAsSpan<T>(string? name = null) where T : unmanaged;
    long[] GetOutputShape(string? name = null);
}
