namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Stores named resources resolved while building a pipeline program.
/// </summary>
public sealed class VmResourceRegistry : IAsyncDisposable, IDisposable
{
    readonly Dictionary<string, IVmResource> resources = new(StringComparer.Ordinal);

    public void Add(IVmResource resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        ArgumentException.ThrowIfNullOrWhiteSpace(resource.Name);

        if(!resources.TryAdd(resource.Name, resource))
            throw new InvalidOperationException($"Pipeline resource '{resource.Name}' is already registered.");
    }

    public bool TryGet<TResource>(string name, out TResource resource)
        where TResource : class, IVmResource
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if(resources.TryGetValue(name, out IVmResource? untypedResource) && untypedResource is TResource typedResource)
        {
            resource = typedResource;
            return true;
        }

        resource = null!;
        return false;
    }

    public TResource Get<TResource>(string name)
        where TResource : class, IVmResource
    {
        if(TryGet(name, out TResource resource))
            return resource;

        throw new KeyNotFoundException($"Pipeline resource '{name}' was not found or has a different type.");
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        foreach(IVmResource resource in resources.Values.Reverse())
        {
            if(resource is IAsyncDisposable asyncDisposable)
                await asyncDisposable.DisposeAsync().ConfigureAwait(false);
            else if(resource is IDisposable disposable)
                disposable.Dispose();
        }

        resources.Clear();
    }
}

