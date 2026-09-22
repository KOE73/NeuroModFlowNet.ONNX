namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Synchronizes one program point by accepted run id.
/// </summary>
/// <remarks>
/// This instruction orders the point of release by accepted <c>RunId</c>: runs are released here in dense RunId order,
/// and later runs are unblocked when an earlier accepted execution failed before reaching this point. It does NOT create
/// a critical section around the following instructions, so by itself it does not serialize a stateful resource — two
/// runs can enter the next instruction concurrently. A stage that needs per-frame mutual exclusion (for example a
/// sequential tracker) must additionally serialize its own execution, or be driven through an ordered critical section
/// that holds the gate across the resource call. See docs/architecture/tracker_and_source_config.ru.md.
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

