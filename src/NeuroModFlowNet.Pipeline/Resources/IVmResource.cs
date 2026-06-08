namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Marker for controller-level resources used by VM instructions.
/// </summary>
/// <remarks>
/// Resources own things that should not be copied into a transaction: model runners, batching queues, trackers, sinks,
/// and long-lived native handles. Instructions call resources; transactions only carry per-run inputs and outputs.
/// </remarks>
public interface IVmResource
{
    string Name { get; }
}

