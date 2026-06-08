using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal sealed class OnnxBatchedRequest<TInput, TOutput>
{
    public OnnxBatchedRequest(VmRunIdentity identity, TInput input)
    {
        Identity = identity;
        Input = input;
        Completion = new TaskCompletionSource<TOutput>(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public VmRunIdentity Identity { get; }

    public TInput Input { get; }

    public TaskCompletionSource<TOutput> Completion { get; }
}

