namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Debug-host-only demand gate preserving the wrapped command's descriptor, device and resource ownership.
///
/// RU: Шлюз запроса только для отладочного хоста; сохраняет descriptor, устройство и владение ресурсами команды.
/// </summary>
internal sealed class DemandPreviewOp(IOp inner, Func<VmRunContext, bool> requested) : IOp, IHasExecutionDevice, IAsyncDisposable
{
    public OpDescriptor Descriptor => inner.Descriptor;
    public bool IsGpuExecution => inner is IHasExecutionDevice { IsGpuExecution: true };
    public string ExecutionDeviceName => inner is IHasExecutionDevice device ? device.ExecutionDeviceName : "CPU";

    public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken) =>
        requested(context) ? inner.ExecuteAsync(context, cancellationToken) : ValueTask.FromResult(OpResult.Continue);

    public async ValueTask DisposeAsync()
    {
        if(inner is IAsyncDisposable asynchronous) await asynchronous.DisposeAsync().ConfigureAwait(false);
        else if(inner is IDisposable disposable) disposable.Dispose();
    }
}
