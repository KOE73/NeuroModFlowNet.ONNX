using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly record struct OrtValueBatchInput(OrtValue Value, int BatchSize, bool OwnsValue) : IDisposable
{
    public void Dispose()
    {
        if(OwnsValue)
            Value.Dispose();
    }
}
