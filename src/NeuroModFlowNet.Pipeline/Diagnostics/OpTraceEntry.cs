namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// One instruction execution sample captured by the VM runtime.
/// </summary>
/// <remarks>
/// EN:
/// <see cref="Index"/> is local to the program that executed the instruction. When a controller runs a chain of
/// programs for one accepted run, <see cref="Program"/> tells which program the sample belongs to.
///
/// RU:
/// <see cref="Index"/> локален для программы, выполнившей инструкцию. Когда контроллер выполняет цепочку программ для
/// одного принятого запуска, <see cref="Program"/> показывает, к какой программе относится запись.
/// </remarks>
public sealed record OpTraceEntry(
    int Index,
    string Name,
    string Operation,
    long ElapsedTicks,
    double ElapsedMilliseconds,
    OpResultKind ResultKind,
    string? JumpLabel,
    IReadOnlyList<VarDebugInfo> VariablesAfter,
    string? Program = null);
