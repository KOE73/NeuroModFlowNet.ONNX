namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Small base class for instructions that only need to provide execution logic and static metadata.
/// </summary>
public abstract class OpBase : IOp
{
    protected OpBase(OpDescriptor descriptor)
    {
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
    }

    public OpDescriptor Descriptor { get; }

    public abstract ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken);
}

