namespace NeuroModFlowNet.Pipeline.Video;

internal sealed class EncodedVideoOutput : IAsyncDisposable
{
    readonly PacketOutputWorker[] workers;

    public EncodedVideoOutput(EncodedVideoStream stream, IReadOnlyList<VideoPacketOutput> outputs, Action<VideoOutputEvent>? notify)
    {
        workers = outputs.Select(output => new PacketOutputWorker(stream, output, notify)).ToArray();
    }

    public void Publish(EncodedVideoPacket packet)
    {
        foreach(PacketOutputWorker worker in workers)
            worker.Publish(packet);
    }

    public ValueTask DisposeAsync() =>
        new(Task.WhenAll(workers.Select(worker => worker.DisposeAsync().AsTask())));
}
