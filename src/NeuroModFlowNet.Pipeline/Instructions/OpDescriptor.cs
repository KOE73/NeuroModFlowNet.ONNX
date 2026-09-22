namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Static metadata used for validation, UI, and future JSON compilation.
/// </summary>
public sealed record OpDescriptor(
    string Name,
    string Operation,
    IReadOnlyList<VarRequirement> Reads,
    IReadOnlyList<VarRequirement> Writes,
    bool HasSideEffects = false,
    string? SyncGate = null)
{
    public static OpDescriptor Create(
        string name,
        string operation,
        IReadOnlyList<VarRequirement>? reads = null,
        IReadOnlyList<VarRequirement>? writes = null,
        bool hasSideEffects = false,
        string? syncGate = null) =>
        new(name, operation, reads ?? [], writes ?? [], hasSideEffects, syncGate);
}
