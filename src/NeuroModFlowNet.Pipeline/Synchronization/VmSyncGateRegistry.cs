namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Stores ordered sync gates for one source controller.
/// </summary>
public sealed class VmSyncGateRegistry
{
    readonly object syncRoot = new();
    readonly Dictionary<string, VmSyncGate> gates = new(StringComparer.Ordinal);
    readonly HashSet<long> closedRunIds = [];

    public VmSyncGate GetOrCreate(string gateName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(gateName);

        lock(syncRoot)
        {
            if(gates.TryGetValue(gateName, out VmSyncGate? gate))
                return gate;

            gate = new VmSyncGate(gateName);
            foreach(long closedRunId in closedRunIds)
                gate.NotifyRunClosed(closedRunId);

            gates.Add(gateName, gate);
            return gate;
        }
    }

    public void NotifyRunClosed(long runId)
    {
        VmSyncGate[] snapshot;

        lock(syncRoot)
        {
            closedRunIds.Add(runId);
            snapshot = gates.Values.ToArray();
        }

        foreach(VmSyncGate gate in snapshot)
            gate.NotifyRunClosed(runId);
    }
}
