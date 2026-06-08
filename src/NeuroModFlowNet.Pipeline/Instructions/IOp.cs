namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// One executable instruction in the VM timeline.
/// </summary>
public interface IOp
{
    OpDescriptor Descriptor { get; }

    ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken);
}

