namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Selects the lab execution mode.
/// </summary>
internal enum LabMode
{
    /// <summary>One camera, one VM controller, running exactly as the original lab.</summary>
    SingleStream,

    /// <summary>One camera fanned out to four emulated VM workers with a shared OBB batch service.</summary>
    MultiStream
}
