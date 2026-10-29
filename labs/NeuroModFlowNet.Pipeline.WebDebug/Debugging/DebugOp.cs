using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Debug-host-only decorator; forwards device validation, descriptor, results and native resource ownership.
///
/// RU: Декоратор только отладочного хоста; сохраняет проверку устройства, descriptor, результаты и владение native-ресурсами.
/// </summary>
internal class DebugOp(IOp inner, CameraDebugSession session, string program, int index) : IOp, IAsyncDisposable
{
    public OpDescriptor Descriptor => inner.Descriptor;
    protected IOp Inner => inner;

    public async ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        // Warmup keeps the existing placement checks but does not stop or take debugger snapshots.
        if(context.Identity.SourceId == session.CameraId + "_warmup")
        {
            session.RecordWarmup(context, program, index, Descriptor.Name);
            return await inner.ExecuteAsync(context, cancellationToken).ConfigureAwait(false);
        }

        await session.BeforeAsync(context, program, index, Descriptor, cancellationToken).ConfigureAwait(false);
        long started = Stopwatch.GetTimestamp();
        try
        {
            OpResult result = await inner.ExecuteAsync(context, cancellationToken).ConfigureAwait(false);
            session.RecordTiming(program, index, Descriptor.Name, Stopwatch.GetElapsedTime(started).TotalMilliseconds);
            await session.AfterAsync(context, program, index, Descriptor, cancellationToken).ConfigureAwait(false);
            return result;
        }
        catch(Exception exception) when(exception is not OperationCanceledException)
        {
            session.Fail(exception);
            throw;
        }
    }

    public async ValueTask DisposeAsync()
    {
        if(inner is IAsyncDisposable asyncDisposable)
            await asyncDisposable.DisposeAsync().ConfigureAwait(false);
        else if(inner is IDisposable disposable)
            disposable.Dispose();
    }
}
