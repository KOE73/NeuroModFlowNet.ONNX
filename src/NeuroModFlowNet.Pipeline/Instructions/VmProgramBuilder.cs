namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Fluent builder for small hand-written VM programs.
/// </summary>
public sealed class VmProgramBuilder
{
    readonly List<IOp> instructions = [];
    readonly Dictionary<string, int> labels = new(StringComparer.Ordinal);
    readonly string? name;

    /// <param name="name">
    /// EN: Optional program name for trace; see <see cref="VmProgram.Name"/>.
    ///
    /// RU: Необязательное имя программы для trace; см. <see cref="VmProgram.Name"/>.
    /// </param>
    public VmProgramBuilder(string? name = null)
    {
        this.name = name;
    }

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
        new(instructions.ToArray(), new Dictionary<string, int>(labels, StringComparer.Ordinal), name);
}

