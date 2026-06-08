namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Collection of named input values used to initialize a single pipeline run context.
/// RU: Коллекция именованных входных значений, используемых для инициализации контекста запуска пайплайна.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Acts as a generic parameters dictionary mapping strings to values with cleanup flags.
/// Reasons: Keeps the VM engine source-independent. Video feeds, files, or message queues can supply different payloads (e.g. frame matrices, timestamps, or flags) without requiring changes to the internal instruction executor or controller APIs.
/// 
/// RU:
/// Суть: Служит универсальным словарем параметров, связывающим строки со значениями и флагами очистки.
/// Причины: Обеспечивает независимость движка VM от источника данных.
///          Видеопотоки, файлы или очереди сообщений могут передавать различные типы данных 
///          (например, матрицы кадров, временные метки или флаги конфигурации) без изменения
///          внутреннего исполнителя инструкций или API контроллера.
/// </remarks>
public sealed class VmRunInputs
{
    readonly Dictionary<string, VmInputValue> values = new(StringComparer.Ordinal);

    public static VmRunInputs Empty { get; } = new();

    public IReadOnlyDictionary<string, VmInputValue> Values => values;

    public VmRunInputs Add<T>(string key, T value, bool disposeWithContext = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        values[key] = new VmInputValue(value, disposeWithContext);
        return this;
    }

    internal void ApplyTo(VmRunContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        foreach(KeyValuePair<string, VmInputValue> item in values)
            context.Set(item.Key, item.Value.Value, item.Value.DisposeWithContext);
    }
}
