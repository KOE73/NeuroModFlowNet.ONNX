namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Decorates an instruction with a runtime condition.
/// </summary>
/// <remarks>
/// This keeps optional branches such as "run OCR only when enabled" outside the model command itself. The wrapped
/// command remains a small domain operation, while the VM timeline still shows where the skipped work would happen.
/// </remarks>
public sealed class OpConditional : IOp
{
    readonly IOp innerInstruction;
    readonly Func<VmRunContext, bool> condition;

    public OpConditional(IOp innerInstruction, Func<VmRunContext, bool> condition)
        : this(innerInstruction, condition, guardName: null)
    {
    }

    public OpConditional(IOp innerInstruction, Func<VmRunContext, bool> condition, string? guardName)
    {
        this.innerInstruction = innerInstruction ?? throw new ArgumentNullException(nameof(innerInstruction));
        this.condition = condition ?? throw new ArgumentNullException(nameof(condition));
        OpDescriptor descriptor = innerInstruction.Descriptor;
        OpControlFlow flow = descriptor.ControlFlow ?? new OpControlFlow();
        Descriptor = descriptor with
        {
            ControlFlow = flow with { Guards = [guardName, .. flow.Guards], FallsThrough = true }
        };
    }

    public OpDescriptor Descriptor { get; }

    public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken) =>
        condition(context)
            ? innerInstruction.ExecuteAsync(context, cancellationToken)
            : ValueTask.FromResult(OpResult.Continue);
}

