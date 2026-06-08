namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Diagnostics collected for one VM run.
/// </summary>
public sealed class VmRunTrace
{
    readonly List<OpTraceEntry> instructions = [];

    public IReadOnlyList<OpTraceEntry> Instructions => instructions;

    public double TotalInstructionMilliseconds => instructions.Sum(instruction => instruction.ElapsedMilliseconds);

    internal void Add(OpTraceEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        instructions.Add(entry);
    }
}
