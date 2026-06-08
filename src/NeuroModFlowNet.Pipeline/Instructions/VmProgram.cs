using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Compiled timeline of VM instructions and labels.
/// </summary>
[DebuggerTypeProxy(typeof(VmProgramDebugView))]
[DebuggerDisplay("Instructions = {Instructions.Count}, Labels = {Labels.Count}")]
public sealed class VmProgram
{
    readonly IReadOnlyList<IOp> instructions;
    readonly IReadOnlyDictionary<string, int> labels;

    public VmProgram(IReadOnlyList<IOp> instructions, IReadOnlyDictionary<string, int> labels)
    {
        this.instructions = instructions ?? throw new ArgumentNullException(nameof(instructions));
        this.labels = labels ?? throw new ArgumentNullException(nameof(labels));
    }

    public static VmProgram Empty { get; } = new([], new Dictionary<string, int>(StringComparer.Ordinal));

    public IReadOnlyList<IOp> Instructions => instructions;

    public IReadOnlyDictionary<string, int> Labels => labels;

    public ValueTask ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken,
        bool captureVariablesInTrace = true)
    {
        return ExecuteLoopInternalAsync(
            context,
            (inst, ctx, ct) => inst.ExecuteAsync(ctx, ct),
            captureVariablesInTrace,
            cancellationToken);
    }

    public ValueTask ExecuteWarmupAsync(
        VmRunContext context,
        CancellationToken cancellationToken,
        bool captureVariablesInTrace = true)
    {
        return ExecuteLoopInternalAsync(context, async (inst, ctx, ct) =>
        {
            var result = await inst.ExecuteAsync(ctx, ct).ConfigureAwait(false);
            if (result.Kind != OpResultKind.Fail)
            {
                DeviceCompatibilityValidator.Validate(inst, ctx);
            }
            return result;
        }, captureVariablesInTrace, cancellationToken);
    }

    private async ValueTask ExecuteLoopInternalAsync(
        VmRunContext context,
        Func<IOp, VmRunContext, CancellationToken, ValueTask<OpResult>> stepExecutor,
        bool captureVariablesInTrace,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        int instructionIndex = 0;
        while(instructionIndex < instructions.Count)
        {
            cancellationToken.ThrowIfCancellationRequested();

            IOp instruction = instructions[instructionIndex];
            if(context.DebugGate is { } debugGate)
            {
                var debugPoint = new OpDebugPoint(
                    context.Identity,
                    instructionIndex,
                    instruction.Descriptor.Name,
                    instruction.Descriptor.Operation,
                    context.SnapshotVariables());
                await debugGate.WaitBeforeInstructionAsync(debugPoint, cancellationToken).ConfigureAwait(false);
            }

            long startTimestamp = Stopwatch.GetTimestamp();
            OpResult result = await stepExecutor(instruction, context, cancellationToken).ConfigureAwait(false);
            long elapsedTicks = Stopwatch.GetTimestamp() - startTimestamp;
            context.Trace.Add(new OpTraceEntry(
                instructionIndex,
                instruction.Descriptor.Name,
                instruction.Descriptor.Operation,
                elapsedTicks,
                elapsedTicks * 1000.0 / Stopwatch.Frequency,
                result.Kind,
                result.Label,
                captureVariablesInTrace ? context.SnapshotVariables() : []));

            instructionIndex = result.Kind switch
            {
                OpResultKind.Continue => instructionIndex + 1,
                OpResultKind.Stop => instructions.Count,
                OpResultKind.Jump => ResolveJump(result.Label),
                OpResultKind.Fail => throw new OpFailureException(
                    instruction.Descriptor.Name,
                    result.Reason ?? "Unknown instruction failure.",
                    result.Exception),
                _ => throw new InvalidOperationException($"Unsupported instruction result: {result.Kind}.")
            };
        }
    }

    int ResolveJump(string? label)
    {
        if(string.IsNullOrWhiteSpace(label))
            throw new InvalidOperationException("Jump instruction did not provide a target label.");

        return labels.TryGetValue(label, out int instructionIndex)
            ? instructionIndex
            : throw new KeyNotFoundException($"Pipeline label '{label}' was not found.");
    }
}

internal sealed record VmProgramDebugView(VmProgram Program)
{
    public string TextTable
    {
        get
        {
            var reverseLabels = new Dictionary<int, List<string>>();
            foreach (var kvp in Program.Labels)
            {
                if (!reverseLabels.TryGetValue(kvp.Value, out var list))
                {
                    list = [];
                    reverseLabels[kvp.Value] = list;
                }
                list.Add(kvp.Key);
            }

            var rows = new List<(string Index, string Labels, string Name, string Operation, string Parameters)>();
            int maxIdxLen = 5;
            int maxLabelLen = 8;
            int maxNameLen = 16;
            int maxOpLen = 9;
            int maxParamsLen = 10;

            for (int i = 0; i < Program.Instructions.Count; i++)
            {
                var inst = Program.Instructions[i];
                var idxStr = i.ToString();
                var labelStr = reverseLabels.TryGetValue(i, out var list) ? string.Join(", ", list) : "";
                var nameStr = inst.Descriptor.Name ?? "";
                var opStr = inst.Descriptor.Operation ?? "";
                var paramsStr = GetInstructionParams(inst);

                maxIdxLen = Math.Max(maxIdxLen, idxStr.Length);
                maxLabelLen = Math.Max(maxLabelLen, labelStr.Length);
                maxNameLen = Math.Max(maxNameLen, nameStr.Length);
                maxOpLen = Math.Max(maxOpLen, opStr.Length);
                maxParamsLen = Math.Max(maxParamsLen, paramsStr.Length);

                rows.Add((idxStr, labelStr, nameStr, opStr, paramsStr));
            }

            var sb = new System.Text.StringBuilder();
            
            // Header
            sb.Append("| ").Append("Index".PadRight(maxIdxLen))
              .Append(" | ").Append("Label(s)".PadRight(maxLabelLen))
              .Append(" | ").Append("Instruction Name".PadRight(maxNameLen))
              .Append(" | ").Append("Operation".PadRight(maxOpLen))
              .Append(" | ").Append("Parameters".PadRight(maxParamsLen))
              .AppendLine(" |");

            sb.Append("|").Append(new string('-', maxIdxLen + 2))
              .Append("|").Append(new string('-', maxLabelLen + 2))
              .Append("|").Append(new string('-', maxNameLen + 2))
              .Append("|").Append(new string('-', maxOpLen + 2))
              .Append("|").Append(new string('-', maxParamsLen + 2))
              .AppendLine("|");

            foreach (var row in rows)
            {
                sb.Append("| ").Append(row.Index.PadRight(maxIdxLen))
                  .Append(" | ").Append(row.Labels.PadRight(maxLabelLen))
                  .Append(" | ").Append(row.Name.PadRight(maxNameLen))
                  .Append(" | ").Append(row.Operation.PadRight(maxOpLen))
                  .Append(" | ").Append(row.Parameters.PadRight(maxParamsLen))
                  .AppendLine(" |");
            }

            return sb.ToString();
        }
    }

    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    public InstructionDebugItem[] Items
    {
        get
        {
            var items = new InstructionDebugItem[Program.Instructions.Count];
            var reverseLabels = new Dictionary<int, string>();
            
            foreach (var kvp in Program.Labels)
            {
                if (reverseLabels.TryGetValue(kvp.Value, out string? existing))
                {
                    reverseLabels[kvp.Value] = existing + ", " + kvp.Key;
                }
                else
                {
                    reverseLabels[kvp.Value] = kvp.Key;
                }
            }

            for (int i = 0; i < Program.Instructions.Count; i++)
            {
                reverseLabels.TryGetValue(i, out string? label);
                var inst = Program.Instructions[i];
                items[i] = new InstructionDebugItem(
                    i,
                    label,
                    inst.Descriptor.Name,
                    inst.Descriptor.Operation,
                    GetInstructionParams(inst),
                    inst);
            }

            return items;
        }
    }

    private static string GetInstructionParams(IOp inst)
    {
        var type = inst.GetType();
        var paramList = new List<string>();

        // Scan fields (both public and private instance)
        var fields = type.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        foreach (var field in fields)
        {
            if (field.Name.StartsWith('<')) // skip backing fields for auto-properties
                continue;

            try
            {
                var val = field.GetValue(inst);
                if (val != null && IsSimpleType(val))
                {
                    paramList.Add($"{field.Name}: {val}");
                }
            }
            catch { }
        }

        // Scan properties (public instance)
        var properties = type.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
        foreach (var prop in properties)
        {
            if (prop.Name == nameof(IOp.Descriptor))
                continue;

            try
            {
                var val = prop.GetValue(inst);
                if (val != null && IsSimpleType(val))
                {
                    bool exists = false;
                    foreach (var existing in paramList)
                    {
                        if (existing.StartsWith(prop.Name + ":", StringComparison.OrdinalIgnoreCase))
                        {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists)
                    {
                        paramList.Add($"{prop.Name}: {val}");
                    }
                }
            }
            catch { }
        }

        return string.Join(", ", paramList);
    }

    private static bool IsSimpleType(object val)
    {
        var t = val.GetType();
        return t.IsPrimitive || 
               val is string || 
               t.IsEnum || 
               val is decimal || 
               val is TimeSpan || 
               val is Guid;
    }
}

[DebuggerDisplay("{DebuggerDisplay,nq}")]
internal readonly record struct InstructionDebugItem(
    int Index,
    string? Label,
    string Name,
    string Operation,
    string Parameters,
    IOp Instruction)
{
    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    private string DebuggerDisplay
    {
        get
        {
            var baseStr = Label != null ? $"[{Label}] {Name} ({Operation})" : $"{Name} ({Operation})";
            return string.IsNullOrEmpty(Parameters) ? baseStr : $"{baseStr} | {Parameters}";
        }
    }
}
