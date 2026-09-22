namespace NeuroModFlowNet.CV.Tracking;

/// <summary>
/// Hot-reloadable per-frame tracker options read by the VM from per-source memory.
/// </summary>
public readonly record struct TrackerFrameOptions(
    TrackRect? StartZone = null,
    TrackRect? EndZone = null);
