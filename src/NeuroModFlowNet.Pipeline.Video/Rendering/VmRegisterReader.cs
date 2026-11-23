namespace NeuroModFlowNet.Pipeline.Video;

internal sealed class VmRegisterReader(VmRunContext context) : IVmRegisterReader
{
    public T Get<T>(string key) => context.Get<T>(key);
    public bool TryGet<T>(string key, out T value) => context.TryGet(key, out value);
}
