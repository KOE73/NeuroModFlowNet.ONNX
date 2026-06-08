namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Coarse lifecycle timing for one controller-managed VM run.
/// </summary>
/// <remarks>
/// Instruction trace measures only time spent inside individual instruction <c>ExecuteAsync</c> calls. This timing
/// explains the rest of the controller lifecycle: output capture, transaction disposal and final bookkeeping.
/// </remarks>
public sealed record VmRunTiming(
    double TotalMilliseconds,
    double ProgramMilliseconds,
    double OutputCaptureMilliseconds,
    double ContextDisposeMilliseconds,
    double CleanupMilliseconds);
