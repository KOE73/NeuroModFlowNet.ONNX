using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Executes one ONNX inference endpoint and writes its typed result into the current VM transaction.
/// </summary>
/// <remarks>
/// This instruction is intentionally domain-blind. It knows which register to read, which register to write, and which
/// endpoint to call. The endpoint may be an inline runner, a reloadable resource, or a batching service that combines
/// requests from several pipeline executions before returning one result per request.
/// </remarks>
public class Model_Inference<TInput, TOutput> : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly IOnnxInferenceEndpoint<TInput, TOutput> endpoint;
    readonly bool disposeOutputWithContext;

    public Model_Inference(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        IOnnxInferenceEndpoint<TInput, TOutput> endpoint,
        bool disposeOutputWithContext = false)
        : base(descriptor)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.endpoint = endpoint ?? throw new ArgumentNullException(nameof(endpoint));
        this.disposeOutputWithContext = disposeOutputWithContext;
    }

    public override async ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        TInput input = context.Get<TInput>(inputKey);
        TOutput output = await endpoint.InferAsync(context.Identity, input, cancellationToken).ConfigureAwait(false);

        // The endpoint decides what the value means. This generic instruction only transfers ownership into the VM
        // context when the specific output contract requires transaction-scoped disposal.
        context.Set(outputKey, output, disposeOutputWithContext);
        return OpResult.Continue;
    }
}
