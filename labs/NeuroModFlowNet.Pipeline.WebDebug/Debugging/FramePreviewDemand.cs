using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Coalesces explicit result-frame requests; only an owned CPU image reaches the encoder.
///
/// RU: Объединяет явные запросы кадра результата; в кодировщик попадает только собственная CPU-картинка.
/// </summary>
internal sealed class FramePreviewDemand : IDisposable
{
    readonly object syncRoot = new();
    TaskCompletionSource<byte[]>? pending;
    Task? encoder;
    int waiters;
    bool claimed, encoding, disposed;
    long prepared, encoded;

    public object Snapshot() => new { prepared = Interlocked.Read(ref prepared), encoded = Interlocked.Read(ref encoded), pending = Volatile.Read(ref waiters) };

    public bool ShouldPrepare(bool observationReady)
    {
        bool requested = observationReady;
        if(Volatile.Read(ref waiters) > 0)
            lock(syncRoot)
                if(pending is not null && !claimed) { claimed = true; requested = true; }
        if(requested) Interlocked.Increment(ref prepared);
        return requested;
    }

    public async Task<byte[]> RequestAsync(CancellationToken cancellationToken)
    {
        TaskCompletionSource<byte[]> request;
        lock(syncRoot)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            request = pending ??= new(TaskCreationOptions.RunContinuationsAsynchronously);
            waiters++;
        }
        try { return await request.Task.WaitAsync(cancellationToken).ConfigureAwait(false); }
        finally
        {
            lock(syncRoot)
            {
                waiters--;
                if(waiters == 0 && pending == request && !encoding) { pending = null; claimed = false; }
            }
        }
    }

    public void Publish(Mat preview)
    {
        lock(syncRoot)
        {
            if(disposed || pending is null || !claimed || encoding) return;
            TaskCompletionSource<byte[]> request = pending;
            try
            {
                Mat owned = preview.Clone();
                encoding = true;
                encoder = Task.Run(() =>
                {
                    byte[]? bytes = null;
                    Exception? failure = null;
                    try
                    {
                        using(owned)
                        {
                            bytes = TensorImagePreviewRenderer.EncodePng(owned);
                            Interlocked.Increment(ref encoded);
                        }
                    }
                    catch(Exception exception) { failure = exception; }
                    lock(syncRoot)
                    {
                        if(pending == request) { pending = null; claimed = false; }
                        encoding = false;
                    }
                    if(failure is not null) request.TrySetException(failure);
                    else request.TrySetResult(bytes!);
                });
            }
            catch(Exception exception) { pending = null; claimed = encoding = false; request.TrySetException(exception); }
        }
    }

    public void Fail(Exception exception)
    {
        lock(syncRoot) { pending?.TrySetException(exception); pending = null; claimed = false; }
    }

    public void Dispose()
    {
        Task? worker;
        lock(syncRoot)
        {
            disposed = true;
            pending?.TrySetException(new InvalidOperationException("The preview session has ended."));
            pending = null;
            worker = encoder;
        }
        worker?.GetAwaiter().GetResult();
    }
}
