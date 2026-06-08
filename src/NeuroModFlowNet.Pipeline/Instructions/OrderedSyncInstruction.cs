namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Synchronizes one program point by accepted run id.
/// </summary>
/// <remarks>
/// Use this before stateful resources such as trackers. The tracker stays focused on tracking; the instruction handles
/// ordering and unblocks later runs when an earlier accepted execution failed before reaching this point.
/// </remarks>
public sealed class OrderedSyncInstruction : OpBase
{
    readonly string gateName;

    public OrderedSyncInstruction(string gateName)
        : base(OpDescriptor.Create($"sync:{gateName}", "sync.ordered", hasSideEffects: true))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(gateName);
        this.gateName = gateName;
    }

    public override async ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        VmSyncGate gate = context.SyncGates.GetOrCreate(gateName);
        await gate.ArriveAndWaitAsync(context.Identity.RunId, cancellationToken).ConfigureAwait(false);
        return OpResult.Continue;
    }
}

