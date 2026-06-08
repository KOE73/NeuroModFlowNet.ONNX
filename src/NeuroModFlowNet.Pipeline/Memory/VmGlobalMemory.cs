using System.Collections.Concurrent;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Shared memory owned by a source controller rather than by one transaction.
/// </summary>
/// <remarks>
/// Tracker state, resource registries, and other controller-level objects belong here. Per-frame tensors and detection
/// results belong in <see cref="VmRunContext"/> so they can be disposed when the accepted run finishes.
/// </remarks>
public sealed class VmGlobalMemory
{
    readonly ConcurrentDictionary<string, object> values = new(StringComparer.Ordinal);

    public VmGlobalMemory()
    {
        Resources = new VmResourceRegistry();
    }

    public VmResourceRegistry Resources { get; }

    public void Set<T>(string key, T value) where T : notnull
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        values[key] = value;
    }

    public bool TryGet<T>(string key, out T value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        if(values.TryGetValue(key, out object? untypedValue) && untypedValue is T typedValue)
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

        throw new KeyNotFoundException($"Global value '{key}' was not found or has a different type.");
    }

    public T GetOrAdd<T>(string key, Func<string, T> factory) where T : notnull
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        ArgumentNullException.ThrowIfNull(factory);

        object value = values.GetOrAdd(key, static (itemKey, state) => state(itemKey)!, factory);
        return value is T typedValue
            ? typedValue
            : throw new InvalidOperationException($"Global value '{key}' already exists with type {value.GetType().Name}.");
    }
}

