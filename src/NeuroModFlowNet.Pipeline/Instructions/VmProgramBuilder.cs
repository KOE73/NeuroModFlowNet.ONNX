namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Fluent builder for small hand-written VM programs.
/// </summary>
public sealed class VmProgramBuilder
{
    readonly List<IOp> instructions = [];
    readonly Dictionary<string, int> labels = new(StringComparer.Ordinal);

    public VmProgramBuilder Label(string label)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(label);

        if(!labels.TryAdd(label, instructions.Count))
            throw new InvalidOperationException($"Pipeline label '{label}' is already registered.");

        return this;
    }

    public VmProgramBuilder Step(IOp instruction)
    {
        ArgumentNullException.ThrowIfNull(instruction);
        instructions.Add(instruction);
        return this;
    }

    public VmProgram Build() =>
        new(instructions.ToArray(), new Dictionary<string, int>(labels, StringComparer.Ordinal));
}

