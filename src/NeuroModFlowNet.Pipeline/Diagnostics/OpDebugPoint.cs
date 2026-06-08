namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Immutable VM stop point captured immediately before an instruction is executed.
/// </summary>
/// <remarks>
/// The debugger sees the transaction state exactly as the next instruction will see it. The point carries metadata and
/// variable diagnostics only; mutable transaction access stays inside the VM runtime.
/// </remarks>
public sealed record OpDebugPoint(
    VmRunIdentity Identity,
    int InstructionIndex,
    string InstructionName,
    string Operation,
    IReadOnlyList<VarDebugInfo> VariablesBefore);
