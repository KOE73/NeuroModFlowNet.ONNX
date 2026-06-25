namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Local named register set for one accepted VM execution.
/// </summary>
/// <remarks>
/// String keys are the external contract because pipeline definitions and scripts are name-driven. A future compiler
/// can still map these names to fast internal slots without changing the user-visible model.
/// </remarks>
public sealed class VmRunContext : IDisposable, IAsyncDisposable
{
    readonly Dictionary<string, object?> values = new(StringComparer.Ordinal);
    readonly List<IDisposable> disposables = [];
    readonly List<IAsyncDisposable> asyncDisposables = [];
    bool disposed;

    public VmRunContext(
        VmRunIdentity identity,
        VmGlobalMemory globalMemory,
        VmSyncGateRegistry syncGates,
        IOpDebugGate? debugGate = null)
    {
        Identity = identity;
        GlobalMemory = globalMemory ?? throw new ArgumentNullException(nameof(globalMemory));
        SyncGates = syncGates ?? throw new ArgumentNullException(nameof(syncGates));
        DebugGate = debugGate;
        Trace = new VmRunTrace();
    }

    public VmRunIdentity Identity { get; }

    public VmGlobalMemory GlobalMemory { get; }

    public VmSyncGateRegistry SyncGates { get; }

    public VmResourceRegistry Resources => GlobalMemory.Resources;

    public VmRunTrace Trace { get; }

    public IOpDebugGate? DebugGate { get; }

    public IReadOnlyCollection<string> Keys => values.Keys;

    public void Set<T>(string key, T value, bool disposeWithContext = false)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        values[key] = value;

        if(disposeWithContext && value is not null)
            AddOwnedResource(value);
    }

    public bool TryGet<T>(string key, out T value)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        if(values.TryGetValue(key, out object? untypedValue) && untypedValue is T typedValue)
        {
            value = typedValue;
            return true;
        }

        value = default!;
        return false;
    }

    public bool TryGetObject(string key, out object? value)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        return values.TryGetValue(key, out value);
    }

    public T Get<T>(string key)
    {
        if(TryGet(key, out T value))
            return value;

        throw new KeyNotFoundException($"Run context value '{key}' was not found or has a different type.");
    }

    public bool Contains(string key)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        return values.ContainsKey(key);
    }

    public bool Remove(string key)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        return values.Remove(key);
    }

    public IReadOnlyList<VarDebugInfo> SnapshotVariables()
    {
        ThrowIfDisposed();

        return values
            .OrderBy(item => item.Key, StringComparer.Ordinal)
            .Select(item => VarDebugInspector.Create(item.Key, item.Value))
            .ToArray();
    }

    public void AddOwnedResource(object resource)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(resource);

        if(resource is IAsyncDisposable asyncDisposable)
            asyncDisposables.Add(asyncDisposable);
        else if(resource is IDisposable disposable)
            disposables.Add(disposable);
        else
            throw new ArgumentException("Owned resources must implement IDisposable or IAsyncDisposable.", nameof(resource));
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        if(disposed)
            return;

        disposed = true;

        for(int index = asyncDisposables.Count - 1; index >= 0; index--)
            await asyncDisposables[index].DisposeAsync().ConfigureAwait(false);

        for(int index = disposables.Count - 1; index >= 0; index--)
            disposables[index].Dispose();

        values.Clear();
        asyncDisposables.Clear();
        disposables.Clear();
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(VmRunContext));
    }
}
