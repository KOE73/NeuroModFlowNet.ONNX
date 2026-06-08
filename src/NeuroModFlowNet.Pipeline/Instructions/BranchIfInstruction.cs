namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Conditional branch instruction for timeline-style programs.
/// </summary>
/// <remarks>
/// Branches should normally read compact CPU metadata, not force large GPU tensors back to CPU just to decide control
/// flow.
/// </remarks>
public sealed class BranchIfInstruction : OpBase
{
    readonly Func<VmRunContext, bool> condition;
    readonly string targetLabel;

    public BranchIfInstruction(
        string name,
        string targetLabel,
        Func<VmRunContext, bool> condition,
        IReadOnlyList<VarRequirement>? reads = null)
        : base(OpDescriptor.Create(name, "branch.if", reads, hasSideEffects: false))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(targetLabel);
        this.condition = condition ?? throw new ArgumentNullException(nameof(condition));
        this.targetLabel = targetLabel;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(condition(context)
            ? OpResult.Jump(targetLabel)
            : OpResult.Continue);
    }
}

