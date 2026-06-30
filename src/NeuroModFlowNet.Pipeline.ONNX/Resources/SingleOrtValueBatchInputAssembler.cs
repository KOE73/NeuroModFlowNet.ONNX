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

        long[] inputShape = inputs[0].GetTensorTypeAndShape().Shape;
        if(inputShape.Length == 0 || inputShape[0] <= 0)
            throw new InvalidOperationException($"Single-input service input must have a positive batch dimension, actual shape: [{string.Join(", ", inputShape)}].");

        int batchSize = checked((int)inputShape[0]);
        return new OrtValueBatchInput(inputs[0], batchSize, OwnsValue: false, RequestItemCounts: [batchSize]);
    }

    public void Dispose()
    {
    }
}
