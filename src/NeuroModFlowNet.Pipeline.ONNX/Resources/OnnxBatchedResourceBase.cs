using System.Threading.Channels;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Base class for ONNX resources that batch requests from parallel VM executions.
/// </summary>
/// <remarks>
/// This is a compute scheduler, not a pipeline queue between modules. A VM execution reaches one instruction, submits
/// a typed request, waits for its own result, and then continues with the same transaction.
/// </remarks>
public abstract class OnnxBatchedResourceBase<TInput, TOutput> :
    IVmResource,
    IOnnxInferenceEndpoint<TInput, TOutput>,
    IAsyncDisposable,
    IDisposable
{
    readonly Channel<OnnxBatchedRequest<TInput, TOutput>> requestChannel;
    readonly CancellationTokenSource shutdownTokenSource = new();
    readonly Task workerTask;
    bool disposed;

    protected OnnxBatchedResourceBase(string name, OnnxBatchedResourceOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        Name = name;
        Options = options;
        requestChannel = Channel.CreateBounded<OnnxBatchedRequest<TInput, TOutput>>(
            new BoundedChannelOptions(options.MaxPendingRequests)
            {
                // One worker owns the runner and the batch assembly state. Many VM executions may submit requests
                // concurrently, but inference itself stays serialized inside this resource.
                SingleReader = true,
                SingleWriter = false,
                FullMode = BoundedChannelFullMode.Wait
            });

        workerTask = Task.Run(() => RunWorkerAsync(shutdownTokenSource.Token));
    }

    public string Name { get; }

    public OnnxBatchedResourceOptions Options { get; }

    public ValueTask<TOutput> InferAsync(
        VmRunIdentity identity,
        TInput input,
        CancellationToken cancellationToken) =>
        ExecuteAsync(identity, input, cancellationToken);

    public async ValueTask<TOutput> ExecuteAsync(VmRunIdentity identity, TInput input, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();

        var request = new OnnxBatchedRequest<TInput, TOutput>(identity, input);

        // Do not await bounded-channel space here. A pipeline run should either enter the inference queue immediately
        // or receive a deterministic overload error; otherwise upstream VM steps can accumulate unbounded latency.
        if(!requestChannel.Writer.TryWrite(request))
            throw new OnnxResourceBusyException(Name);

        // Cancellation belongs to the waiting transaction, not to the whole batching worker. If one caller gives up, the
        // worker can keep serving other requests and TrySet* below will safely ignore this canceled completion.
        using CancellationTokenRegistration registration = cancellationToken.Register(
            static state => ((OnnxBatchedRequest<TInput, TOutput>)state!).Completion.TrySetCanceled(),
            request);

        return await request.Completion.Task.ConfigureAwait(false);
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        if(disposed)
            return;

        disposed = true;
        requestChannel.Writer.TryComplete();
        await shutdownTokenSource.CancelAsync().ConfigureAwait(false);

        try
        {
            await workerTask.ConfigureAwait(false);
        }
        catch(OperationCanceledException)
        {
        }

        shutdownTokenSource.Dispose();
        DisposeCore();
    }

    protected virtual void DisposeCore()
    {
    }

    protected abstract ValueTask<IReadOnlyList<TOutput>> ExecuteBatchAsync(
        IReadOnlyList<TInput> inputs,
        CancellationToken cancellationToken);

    async Task RunWorkerAsync(CancellationToken cancellationToken)
    {
        var batch = new List<OnnxBatchedRequest<TInput, TOutput>>(Options.MaxBatchSize);

        while(await requestChannel.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
        {
            batch.Clear();
            if(!requestChannel.Reader.TryRead(out OnnxBatchedRequest<TInput, TOutput>? firstRequest))
                continue;

            // The first request starts a latency window. Additional requests are optional: they improve throughput only
            // while MaxWaitTime allows it.
            batch.Add(firstRequest);
            await FillBatchAsync(batch, cancellationToken).ConfigureAwait(false);
            await ExecuteBatchAndCompleteRequestsAsync(batch, cancellationToken).ConfigureAwait(false);
        }
    }

    async ValueTask FillBatchAsync(List<OnnxBatchedRequest<TInput, TOutput>> batch, CancellationToken cancellationToken)
    {
        while(batch.Count < Options.MaxBatchSize && requestChannel.Reader.TryRead(out OnnxBatchedRequest<TInput, TOutput>? request))
            batch.Add(request);

        TimeSpan maxWaitTime = Options.EffectiveMaxWaitTime;
        if(batch.Count >= Options.MaxBatchSize || maxWaitTime == TimeSpan.Zero)
            return;

        // The delay is canceled after the batch is ready so the timer task does not survive into the next batch cycle.
        using var waitCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task delayTask = Task.Delay(maxWaitTime, waitCancellation.Token);

        while(batch.Count < Options.MaxBatchSize)
        {
            Task<bool> waitTask = requestChannel.Reader.WaitToReadAsync(cancellationToken).AsTask();
            Task completedTask = await Task.WhenAny(waitTask, delayTask).ConfigureAwait(false);

            if(completedTask == delayTask || !await waitTask.ConfigureAwait(false))
                break;

            // Drain all currently queued requests after each wakeup. This avoids one await per request during bursty
            // camera/debug workloads and keeps batch construction deterministic.
            while(batch.Count < Options.MaxBatchSize && requestChannel.Reader.TryRead(out OnnxBatchedRequest<TInput, TOutput>? request))
                batch.Add(request);
        }

        await waitCancellation.CancelAsync().ConfigureAwait(false);
    }

    async ValueTask ExecuteBatchAndCompleteRequestsAsync(
        IReadOnlyList<OnnxBatchedRequest<TInput, TOutput>> requests,
        CancellationToken cancellationToken)
    {
        try
        {
            // ExecuteBatchAsync receives only model inputs; request metadata stays in the scheduler so outputs can be
            // mapped back to the exact waiting transactions in original queue order.
            TInput[] inputs = requests.Select(request => request.Input).ToArray();
            IReadOnlyList<TOutput> outputs = await ExecuteBatchAsync(inputs, cancellationToken).ConfigureAwait(false);

            if(outputs.Count != requests.Count)
                throw new InvalidOperationException($"Resource '{Name}' returned {outputs.Count} outputs for {requests.Count} requests.");

            for(int index = 0; index < requests.Count; index++)
                requests[index].Completion.TrySetResult(outputs[index]);
        }
        catch(Exception exception)
        {
            foreach(OnnxBatchedRequest<TInput, TOutput> request in requests)
                request.Completion.TrySetException(exception);
        }
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(GetType().Name);
    }
}
