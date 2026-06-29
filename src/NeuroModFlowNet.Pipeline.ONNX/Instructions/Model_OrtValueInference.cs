using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Model_OrtValueInference<TOutput> : OnnxBatchedInference<OrtValue, TOutput[]>
{
    public Model_OrtValueInference(
        string name,
        string inputKey,
        string outputKey,
        OrtValueBatchedInferenceEndpoint<TOutput> endpoint)
        : base(
            OpDescriptor.Create(
                name,
                "model.ortValueInference",
                [VarRequirement.Read<OrtValue>(inputKey)],
                [VarRequirement.Write<TOutput[]>(outputKey)],
                hasSideEffects: true),
            inputKey,
            outputKey,
            endpoint)
    {
    }
}
