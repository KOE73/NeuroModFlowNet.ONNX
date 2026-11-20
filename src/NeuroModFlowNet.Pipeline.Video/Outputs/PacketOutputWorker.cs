using System.Diagnostics;
using System.Threading.Channels;

namespace NeuroModFlowNet.Pipeline.Video;

internal sealed class PacketOutputWorker : IAsyncDisposable
{
    readonly EncodedVideoStream stream;
    readonly VideoPacketOutput output;
    readonly Action<VideoOutputEvent>? notify;
    readonly object gate = new();
    readonly Queue<(EncodedVideoPacket Packet, long AddedAt, long Epoch)> queue = new();
    readonly Channel<bool> changed = Channel.CreateBounded<bool>(new BoundedChannelOptions(1)
    {
        FullMode = BoundedChannelFullMode.DropWrite,
        SingleReader = true
    });
    readonly CancellationTokenSource stop = new();
    readonly Task worker;
    bool accepting = true;
    bool awaitingRandomAccess = true;
    long bytes;
    long epoch;
    string? overflow;

    public PacketOutputWorker(EncodedVideoStream stream, VideoPacketOutput output, Action<VideoOutputEvent>? notify)
    {
        this.stream = stream;
        this.output = output;
        this.notify = notify;
        worker = Task.Run(RunAsync);
    }

    #region Submission and cleanup

    public void Publish(EncodedVideoPacket packet)
    {
        lock(gate)
        {
            if(!accepting || (awaitingRandomAccess && !packet.IsRandomAccess))
                return;

            if(queue.Count >= output.MaximumQueuePackets || bytes + packet.Size > output.MaximumQueueBytes ||
                (queue.TryPeek(out var oldest) && HasExpired(oldest.AddedAt)))
            {
                overflow = "The encoded packet queue reached its size or age limit.";
                ResetQueue();
                if(!output.Reconnect)
                    accepting = false;
                changed.Writer.TryWrite(true);
                if(!accepting || !packet.IsRandomAccess || packet.Size > output.MaximumQueueBytes)
                    return;
            }

            queue.Enqueue((packet.Retain(), Stopwatch.GetTimestamp(), epoch));
            bytes += packet.Size;
            awaitingRandomAccess = false;
        }
        changed.Writer.TryWrite(true);
    }

    public async ValueTask DisposeAsync()
    {
        lock(gate)
            accepting = false;
        changed.Writer.TryComplete();
        stop.CancelAfter(TimeSpan.FromSeconds(15));
        try
        {
            await worker.WaitAsync(TimeSpan.FromSeconds(20)).ConfigureAwait(false);
            stop.Dispose();
        }
        catch(TimeoutException)
        {
            lock(gate) ResetQueue();
            Report("stop-timeout", "The destination did not return from its last operation.");
            // A blocked destination owns only its in-flight compressed packet and sink.
            _ = worker.ContinueWith(completed =>
            {
                if(completed.Exception is { } exception)
                    Trace.TraceError(exception.ToString());
                stop.Dispose();
            }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
        }
    }

    void ResetQueue()
    {
        while(queue.TryDequeue(out var item))
            item.Packet.Dispose();
        bytes = 0;
        epoch++;
        awaitingRandomAccess = true;
    }

    bool HasExpired(long timestamp) =>
        Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds > output.MaximumQueueMilliseconds;

    #endregion

    #region Output worker

    bool TryTake(out (EncodedVideoPacket Packet, long AddedAt, long Epoch) item, out string? gap)
    {
        lock(gate)
        {
            if(queue.TryPeek(out var oldest) && HasExpired(oldest.AddedAt))
            {
                overflow = "Queued encoded packets expired.";
                ResetQueue();
                if(!output.Reconnect)
                    accepting = false;
            }
            gap = overflow;
            overflow = null;
            if(!queue.TryDequeue(out item))
                return false;
            bytes -= item.Packet.Size;
            return true;
        }
    }

    async Task RunAsync()
    {
        IVideoPacketSink? sink = null;
        long openEpoch = -1;
        CancellationToken cancellationToken = stop.Token;
        try
        {
            while(true)
            {
                bool hasPacket = TryTake(out var item, out string? gap);
                if(gap is not null)
                {
                    Report("gap", gap);
                    if(!output.Reconnect)
                        break;
                }
                if(!hasPacket)
                {
                    if(!await changed.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
                        break;
                    changed.Reader.TryRead(out _);
                    continue;
                }

                using EncodedVideoPacket packet = item.Packet;
                try
                {
                    if(openEpoch != item.Epoch)
                    {
                        Close(ref sink);
                        sink = output.CreateSink();
                        sink.Open(stream, cancellationToken);
                        openEpoch = item.Epoch;
                        Report("open");
                    }
                    sink!.Write(packet, cancellationToken);
                }
                catch(Exception exception) when(!cancellationToken.IsCancellationRequested)
                {
                    Report(output.Reconnect ? "retry" : "failed", exception.GetBaseException().Message);
                    Close(ref sink);
                    openEpoch = -1;
                    lock(gate)
                    {
                        ResetQueue();
                        if(!output.Reconnect)
                            accepting = false;
                    }
                    if(!output.Reconnect)
                        break;
                    await Task.Delay(TimeSpan.FromMilliseconds(500), cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested) { }
        catch(Exception exception)
        {
            Report("failed", exception.GetBaseException().Message);
        }
        finally
        {
            lock(gate)
            {
                accepting = false;
                ResetQueue();
            }
            Close(ref sink);
            Report("closed");
        }
    }

    void Close(ref IVideoPacketSink? sink)
    {
        IVideoPacketSink? current = sink;
        sink = null;
        try { current?.Dispose(); }
        catch(Exception exception) { Report("finalize-failed", exception.GetBaseException().Message); }
    }

    void Report(string state, string? detail = null)
    {
        try { notify?.Invoke(new(stream.StreamId, output.Id, state, detail)); }
        catch(Exception exception) { Trace.TraceError(exception.ToString()); }
    }

    #endregion
}
