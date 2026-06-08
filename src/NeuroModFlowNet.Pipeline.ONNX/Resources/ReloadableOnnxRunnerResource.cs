using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// ONNX runner resource whose inner runner can be replaced while the VM program keeps the same instruction instance.
/// </summary>
/// <remarks>
/// PaddleOCR recognition changes input/output persistent shapes when batch size or recognition width changes. Rebuilding
/// the whole VM program for that would mix UI tuning with program structure, so only the resource swaps its runner under
/// the same execution lock.
/// </remarks>
public sealed class ReloadableOnnxRunnerResource<TInput, TOutput> :
    IVmResource,
    IOnnxInferenceEndpoint<TInput, TOutput>,
    IDisposable
{
    readonly SemaphoreSlim executionLock = new(1, 1);
    readonly bool disposeReplacedRunner;
    IRunner<TInput, TOutput> runner;
    bool disposed;

    public ReloadableOnnxRunnerResource(
        string name,
        IRunner<TInput, TOutput> runner,
        bool disposeReplacedRunner = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        this.runner = runner ?? throw new ArgumentNullException(nameof(runner));
        this.disposeReplacedRunner = disposeReplacedRunner;
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

        // The same lock protects prediction and replacement. That guarantees a VM run never observes a runner halfway
        // through a shape/model swap, and the old runner is not disposed while Predict is still using native state.
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

    public async ValueTask ReplaceRunnerAsync(IRunner<TInput, TOutput> replacementRunner, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(replacementRunner);
        ThrowIfDisposed();

        await executionLock.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            // Swap first, then dispose the previous runner. If disposal is expensive or throws in the future, the
            // resource already points at the replacement and no caller can continue using the stale runner.
            IRunner<TInput, TOutput> oldRunner = runner;
            runner = replacementRunner;

            // Some current runners own the shared OnnxExecutionContext. PaddleOCR Rec rebuilds the runner when the UI
            // changes persistent shapes, but it must keep that context alive until the whole resource is disposed.
            if(disposeReplacedRunner)
                oldRunner.Dispose();
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
        executionLock.Dispose();
        runner.Dispose();
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(ReloadableOnnxRunnerResource<TInput, TOutput>));
    }
}
