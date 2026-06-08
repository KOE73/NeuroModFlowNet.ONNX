namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Lightweight description of one transaction variable for VM debugging.
/// </summary>
public sealed record VarDebugInfo(
    string Key,
    string TypeName,
    VarMemoryLocation MemoryLocation,
    bool IsDisposable,
    bool IsAsyncDisposable,
    string? Shape = null);
