using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Serializes access to one existing <see cref="IRunner{TIn,TOut}"/> instance.
/// </summary>
/// <remarks>
/// The current ONNX runner owns one <c>OnnxExecutionContext</c> with prepared inputs/outputs. Until OrtValue ownership and
/// IO binding are redesigned, a shared runner must be called through a resource lock rather than directly from several
/// parallel VM executions.
/// </remarks>
public sealed class OnnxRunnerResource<TInput, TOutput> :
    IVmResource,
    IOnnxInferenceEndpoint<TInput, TOutput>,
    IDisposable
{
    readonly IRunner<TInput, TOutput> runner;
    readonly SemaphoreSlim executionLock = new(1, 1);
    bool disposed;

    public OnnxRunnerResource(string name, IRunner<TInput, TOutput> runner)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        this.runner = runner ?? throw new ArgumentNullException(nameof(runner));
    }

    public string Name { get; }

    public ValueTask<TOutput> InferAsync(
        VmRunIdentity identity,
        TInput input,
        CancellationToken cancellationToken) =>
        ExecuteAsync(input, cancellationToken);

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();

        // Most current runners keep mutable OrtValue/IoBinding state inside OnnxExecutionContext. The resource lock makes
        // that state single-consumer even when several VM runs reach the same model command at the same time.
        await executionLock.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return runner.Predict(input);
        }
        finally
        {
            executionLock.Release();
        }
    }

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;

        // Disposal happens after the owning program/resource set is no longer scheduling work. The semaphore is disposed
        // together with the runner to make accidental late use fail loudly through ThrowIfDisposed.
        executionLock.Dispose();
        runner.Dispose();
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(OnnxRunnerResource<TInput, TOutput>));
    }
}
