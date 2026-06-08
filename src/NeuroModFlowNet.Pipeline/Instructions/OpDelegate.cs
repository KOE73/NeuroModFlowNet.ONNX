namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Instruction backed by a delegate, useful for tests, labs, and early custom steps.
/// </summary>
public sealed class OpDelegate : OpBase
{
    readonly Func<VmRunContext, CancellationToken, ValueTask<OpResult>> execute;

    public OpDelegate(
        OpDescriptor descriptor,
        Func<VmRunContext, CancellationToken, ValueTask<OpResult>> execute)
        : base(descriptor)
    {
        this.execute = execute ?? throw new ArgumentNullException(nameof(execute));
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken) =>
        execute(context, cancellationToken);
}

