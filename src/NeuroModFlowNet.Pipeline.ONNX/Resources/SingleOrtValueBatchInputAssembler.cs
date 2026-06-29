using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class SingleOrtValueBatchInputAssembler : IOrtValueBatchInputAssembler
{
    public OrtValueBatchInput Assemble(
        IReadOnlyList<OrtValue> inputs,
        InferenceBackend executionBackend,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();

        if(inputs.Count != 1)
            throw new InvalidOperationException($"Single-input service received {inputs.Count} inputs. Use a batch assembler for batched execution.");

        return new OrtValueBatchInput(inputs[0], BatchSize: 1, OwnsValue: false);
    }

    public void Dispose()
    {
    }
}
