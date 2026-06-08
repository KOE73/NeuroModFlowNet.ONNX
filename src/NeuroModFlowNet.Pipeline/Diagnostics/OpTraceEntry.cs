namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// One instruction execution sample captured by the VM runtime.
/// </summary>
public sealed record OpTraceEntry(
    int Index,
    string Name,
    string Operation,
    long ElapsedTicks,
    double ElapsedMilliseconds,
    OpResultKind ResultKind,
    string? JumpLabel,
    IReadOnlyList<VarDebugInfo> VariablesAfter);
