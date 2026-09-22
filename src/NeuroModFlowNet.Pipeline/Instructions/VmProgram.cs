using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN:
/// Immutable compiled VM program: ordered operations plus a label table for branch targets.
///
/// RU:
/// Неизменяемая скомпилированная VM-программа: упорядоченные операции и таблица меток для переходов.
/// </summary>
/// <remarks>
/// EN:
/// The program owns only instruction order and label resolution. Run-local registers, trace entries, debug gates and
/// disposable outputs belong to <see cref="VmRunContext"/>.
///
/// RU:
/// Программа владеет только порядком инструкций и разрешением меток. Локальные регистры запуска, trace, debug gate и
/// disposable-результаты принадлежат <see cref="VmRunContext"/>.
/// </remarks>
[DebuggerTypeProxy(typeof(VmProgramDebugView))]
[DebuggerDisplay("Name = {Name}, Instructions = {Instructions.Count}, Labels = {Labels.Count}")]
public sealed class VmProgram
{
    readonly IReadOnlyList<IOp> instructions;
    readonly IReadOnlyDictionary<string, int> labels;

    /// <summary>
    /// EN:
    /// Creates a compiled VM program from ordered operations and a label-to-instruction map.
    ///
    /// RU:
    /// Создает скомпилированную VM-программу из упорядоченных операций и карты метка -> индекс инструкции.
    /// </summary>
    /// <param name="name">
    /// EN: Optional program name shown in trace entries; useful when a controller executes a chain of programs.
    ///
    /// RU: Необязательное имя программы для trace; полезно, когда контроллер выполняет цепочку программ.
    /// </param>
    public VmProgram(IReadOnlyList<IOp> instructions, IReadOnlyDictionary<string, int> labels, string? name = null)
    {
        this.instructions = instructions ?? throw new ArgumentNullException(nameof(instructions));
        this.labels = labels ?? throw new ArgumentNullException(nameof(labels));
        Name = name;
    }

    /// <summary>
    /// EN:
    /// Empty program that completes without touching the run context.
    ///
    /// RU:
    /// Пустая программа, которая завершается без изменения контекста запуска.
    /// </summary>
    public static VmProgram Empty { get; } = new([], new Dictionary<string, int>(StringComparer.Ordinal), "empty");

    /// <summary>
    /// EN:
    /// Optional program name. Labels stay local to the program; the name only identifies the program in trace and
    /// diagnostics when a controller runs several programs for one accepted run.
    ///
    /// RU:
    /// Необязательное имя программы. Метки остаются локальными для программы; имя только идентифицирует программу в
    /// trace и диагностике, когда контроллер выполняет несколько программ для одного принятого запуска.
    /// </summary>
    public string? Name { get; }

    /// <summary>
    /// EN:
    /// Ordered VM operations executed from index zero unless a branch changes the next instruction.
    ///
    /// RU:
    /// Упорядоченные VM-операции, выполняемые с нулевого индекса, если переход не изменит следующую инструкцию.
    /// </summary>
    public IReadOnlyList<IOp> Instructions => instructions;

    /// <summary>
    /// EN:
    /// Named branch targets used by <see cref="OpResultKind.Jump"/> results.
    ///
    /// RU:
    /// Именованные цели переходов, используемые результатами <see cref="OpResultKind.Jump"/>.
    /// </summary>
    public IReadOnlyDictionary<string, int> Labels => labels;

    /// <summary>
    /// EN:
    /// Executes the program with the normal step contract: one VM step is one operation execution.
    /// Returns <see cref="VmProgramExit.Stopped"/> when an instruction ended the whole run with
    /// <see cref="OpResultKind.Stop"/>, so a controller can skip the remaining programs of a chain.
    ///
    /// RU:
    /// Выполняет программу с обычным контрактом шага: один VM-шаг равен одному выполнению операции.
    /// Возвращает <see cref="VmProgramExit.Stopped"/>, если инструкция завершила весь запуск через
    /// <see cref="OpResultKind.Stop"/>, чтобы контроллер мог пропустить оставшиеся программы цепочки.
    /// </summary>
    public ValueTask<VmProgramExit> ExecuteAsync(
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

    /// <summary>
    /// EN:
    /// Executes the program as a warmup pass and validates device placement after each successful operation.
    ///
    /// RU:
    /// Выполняет программу как warmup-проход и проверяет размещение данных после каждой успешной операции.
    /// </summary>
    public ValueTask<VmProgramExit> ExecuteWarmupAsync(
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

    /// <summary>
    /// EN:
    /// Runs the shared instruction loop for all execution modes.
    ///
    /// RU:
    /// Выполняет общий цикл инструкций для всех режимов запуска.
    /// </summary>
    /// <param name="context">
    /// EN: Run-local registers, resources, trace and optional debug gate.
    ///
    /// RU: Локальные регистры запуска, ресурсы, trace и необязательный debug gate.
    /// </param>
    /// <param name="stepExecutor">
    /// EN:
    /// Defines what "execute one VM step" means for the current mode. Normal execution passes the direct operation call:
    /// <see cref="IOp.ExecuteAsync(VmRunContext, CancellationToken)"/>. Warmup execution passes the same operation call
    /// wrapped with post-step device compatibility validation. This delegate keeps mode-specific step behavior outside
    /// the instruction loop, so branching, tracing, debug-gate waiting and failure handling remain implemented once.
    ///
    /// RU:
    /// Определяет, что именно означает "выполнить один VM-шаг" для текущего режима. Обычный запуск передает прямой
    /// вызов операции: <see cref="IOp.ExecuteAsync(VmRunContext, CancellationToken)"/>. Warmup-запуск передает тот же
    /// вызов, обернутый проверкой совместимости размещения данных после шага. Этот delegate выносит режимную семантику
    /// шага из цикла инструкций, чтобы переходы, trace, ожидание debug gate и обработка ошибок оставались реализованы
    /// в одном месте.
    /// </param>
    /// <param name="captureVariablesInTrace">
    /// EN: When true, stores a variable snapshot after each operation.
    ///
    /// RU: Если true, сохраняет snapshot переменных после каждой операции.
    /// </param>
    /// <param name="cancellationToken">
    /// EN: Cancellation token used by debug waits and operation execution.
    ///
    /// RU: Токен отмены, используемый ожиданиями debug gate и выполнением операций.
    /// </param>
    private async ValueTask<VmProgramExit> ExecuteLoopInternalAsync(
        VmRunContext context,
        Func<IOp, VmRunContext, CancellationToken, ValueTask<OpResult>> stepExecutor,
        bool captureVariablesInTrace,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        VmProgramExit exit = VmProgramExit.Completed;
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
            OpResult result = await ExecuteStepWithOptionalSyncGateAsync(
                instruction,
                context,
                stepExecutor,
                cancellationToken).ConfigureAwait(false);
            long elapsedTicks = Stopwatch.GetTimestamp() - startTimestamp;
            context.Trace.Add(new OpTraceEntry(
                instructionIndex,
                instruction.Descriptor.Name,
                instruction.Descriptor.Operation,
                elapsedTicks,
                elapsedTicks * 1000.0 / Stopwatch.Frequency,
                result.Kind,
                result.Label,
                captureVariablesInTrace ? context.SnapshotVariables() : [],
                Name));

            if(result.Kind == OpResultKind.Stop)
                exit = VmProgramExit.Stopped;

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

        return exit;
    }

    static async ValueTask<OpResult> ExecuteStepWithOptionalSyncGateAsync(
        IOp instruction,
        VmRunContext context,
        Func<IOp, VmRunContext, CancellationToken, ValueTask<OpResult>> stepExecutor,
        CancellationToken cancellationToken)
    {
        string? syncGateName = instruction.Descriptor.SyncGate;
        if(string.IsNullOrWhiteSpace(syncGateName))
            return await stepExecutor(instruction, context, cancellationToken).ConfigureAwait(false);

        VmSyncGate gate = context.SyncGates.GetOrCreate(syncGateName);
        bool acquired = false;
        try
        {
            await gate.AcquireInOrderAsync(context.Identity.RunId, cancellationToken).ConfigureAwait(false);
            acquired = true;
            return await stepExecutor(instruction, context, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            if(acquired)
                gate.Release(context.Identity.RunId);
        }
    }

    /// <summary>
    /// EN:
    /// Converts a VM label into the instruction index used by the next loop iteration.
    ///
    /// RU:
    /// Преобразует VM-метку в индекс инструкции для следующей итерации цикла.
    /// </summary>
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
    /// <summary>
    /// EN:
    /// Builds a debugger-friendly table with indexes, labels, operations and simple operation parameters.
    ///
    /// RU:
    /// Строит удобную для отладчика таблицу с индексами, метками, операциями и простыми параметрами операций.
    /// </summary>
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

    /// <summary>
    /// EN:
    /// Exposes instructions as debugger root items so the compiled program can be inspected without expanding fields.
    ///
    /// RU:
    /// Показывает инструкции как корневые элементы отладчика, чтобы программу можно было смотреть без раскрытия полей.
    /// </summary>
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

    /// <summary>
    /// EN:
    /// Reads simple field and property values from an operation for debugger display only.
    ///
    /// RU:
    /// Читает простые поля и свойства операции только для отображения в отладчике.
    /// </summary>
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

    /// <summary>
    /// EN:
    /// Checks whether a reflected value is safe and compact enough to show in the debugger table.
    ///
    /// RU:
    /// Проверяет, что отраженное значение безопасно и достаточно компактно для таблицы отладчика.
    /// </summary>
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
    /// <summary>
    /// EN:
    /// Formats one operation row for the debugger variable window.
    ///
    /// RU:
    /// Форматирует одну строку операции для окна переменных отладчика.
    /// </summary>
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
