using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public interface IOrtValueBatchInputAssembler : IDisposable
{
    OrtValueBatchInput Assemble(
        IReadOnlyList<OrtValue> inputs,
        InferenceBackend executionBackend,
        CancellationToken cancellationToken);
}
