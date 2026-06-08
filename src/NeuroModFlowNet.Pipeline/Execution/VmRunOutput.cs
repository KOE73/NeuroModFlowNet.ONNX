namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: DTO containing selected VM output variables captured before run context disposal.
/// RU: DTO с выбранными выходными переменными VM, зафиксированными перед освобождением контекста запуска.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Acts as a read-only container holding variables requested by the host for output.
/// Reasons: Encapsulates execution results. If an output object needs to survive context disposal (e.g. results, bounding boxes), the instruction must ensure it is not registered as context-owned or clone it beforehand.
/// 
/// RU:
/// Суть: Является контейнером только для чтения, хранящим переменные, запрошенные хостом для вывода.
/// Причины: Инкапсулирует результаты выполнения. Если какой-либо выходной объект должен продолжить 
///          существование после удаления контекста запуска (например, результаты детекции), инструкция обязана убедиться,
///          что он не зарегистрирован как принадлежащий контексту, либо клонировать его заранее.
/// </remarks>
public sealed class VmRunOutput
{
    public static VmRunOutput Empty { get; } = new(new Dictionary<string, object?>(StringComparer.Ordinal));

    public VmRunOutput(IReadOnlyDictionary<string, object?> values)
    {
        Values = values ?? throw new ArgumentNullException(nameof(values));
    }

    public IReadOnlyDictionary<string, object?> Values { get; }

    public IEnumerable<string> Keys => Values.Keys;

    public bool TryGet<T>(string key, out T value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        if(Values.TryGetValue(key, out object? untypedValue) && untypedValue is T typedValue)
        {
            value = typedValue;
            return true;
        }

        value = default!;
        return false;
    }

    public T Get<T>(string key)
    {
        if(TryGet(key, out T value))
            return value;

        throw new KeyNotFoundException($"VM output value '{key}' was not found or has a different type.");
    }

    internal static VmRunOutput Capture(VmRunContext context, IReadOnlyCollection<string>? outputKeys)
    {
        ArgumentNullException.ThrowIfNull(context);

        if(outputKeys is null || outputKeys.Count == 0)
            return Empty;

        var values = new Dictionary<string, object?>(outputKeys.Count, StringComparer.Ordinal);
        foreach(string outputKey in outputKeys)
        {
            if(context.TryGetObject(outputKey, out object? value))
                values.Add(outputKey, value);
        }

        return values.Count == 0 ? Empty : new VmRunOutput(values);
    }
}
