using System.Buffers.Binary;
using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class RegisterVideo : IDisposable
{
    static readonly SemaphoreSlim EncoderSlot = new(1);
    Feed? active;
    readonly CancellationTokenSource lifetime = new();

    public void Capture(VmRunContext context, string program, int index, string phase)
    {
        Feed? feed = Volatile.Read(ref active);
        if(feed is not null && feed.Target.Program == program && feed.Target.Index == index && feed.Target.Phase == phase)
            feed.Capture(context);
    }

    public async Task StreamAsync(HttpContext http, string libraries, DebugWatchTarget target)
    {
        if(!EncoderSlot.Wait(0)) throw new InvalidOperationException("An H.264 observation is already open.");
        try
        {
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(lifetime.Token, http.RequestAborted);
            using var feed = new Feed(libraries, target);
            Volatile.Write(ref active, feed);
            try
            {
                http.Response.ContentType = "application/octet-stream";
                await http.Response.StartAsync(stop.Token);
                await feed.SendAsync(http.Response.Body, stop.Token);
            }
            catch(OperationCanceledException) when(stop.IsCancellationRequested) { }
            finally { Volatile.Write(ref active, null); }
        }
        finally { EncoderSlot.Release(); }
    }

    public void Dispose()
    {
        lifetime.Cancel();
        lifetime.Dispose();
    }

    sealed class Feed(string libraries, DebugWatchTarget target) : IDisposable
    {
        CudaH264Encoder? encoder;
        public DebugWatchTarget Target => target;
        readonly Channel<(CudaH264Encoder.Frame Frame, long RunId)> queue = Channel.CreateBounded<(CudaH264Encoder.Frame, long)>(
            new BoundedChannelOptions(3) { SingleReader = true, SingleWriter = true });
        readonly SemaphoreSlim surfaces = new(3);
        readonly object captureGate = new();
        readonly Dictionary<long, long> pending = [];
        long seen, skipped, encoded, captureTicks, encodeTicks;
        long lastTimestamp = -1;
        readonly long started = Stopwatch.GetTimestamp();
        bool stopped;

        public void Capture(VmRunContext context)
        {
            Interlocked.Increment(ref seen);
            if(!Monitor.TryEnter(captureGate)) { Interlocked.Increment(ref skipped); return; }
            try
            {
                if(stopped || !surfaces.Wait(0)) { Interlocked.Increment(ref skipped); return; }
                CudaH264Encoder.Frame? frame = null;
                long timestamp = Stopwatch.GetTimestamp();
                try
                {
                    if(!context.TryGet(target.Key, out OrtValue image))
                        throw new InvalidOperationException($"Register '{target.Key}' is not present as an OrtValue at the selected boundary.");
                    var info = image.GetTensorTypeAndShape();
                    using OrtMemoryInfo placement = image.GetTensorMemoryInfo();
                    if(info.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.UInt8 ||
                        info.Shape.Length != 4 || info.Shape[0] != 1 || info.Shape[3] != 3 || placement.Name != "Cuda" || placement.Id != 0)
                        throw new NotSupportedException("H.264 requires a CUDA:0 BGR U8 NHWC tensor [1,H,W,3].");
                    encoder ??= new(libraries, checked((int)info.Shape[2]), checked((int)info.Shape[1]));
                    long runId = context.Identity.RunId;
                    frame = encoder.Capture(image);
                    if(!queue.Writer.TryWrite((frame, runId))) throw new InvalidOperationException("Video queue is closed.");
                    frame = null;
                }
                catch(Exception error)
                {
                    frame?.Dispose();
                    surfaces.Release();
                    stopped = true;
                    queue.Writer.TryComplete(error);
                }
                finally { Interlocked.Add(ref captureTicks, Stopwatch.GetTimestamp() - timestamp); }
            }
            finally { Monitor.Exit(captureGate); }
        }

        public async Task SendAsync(Stream output, CancellationToken cancellationToken)
        {
            // Little-endian packet size, keyframe flag, PTS, RunId, and five cumulative counters.
            var header = new byte[61];
            var packets = new List<(long Timestamp, long RunId, bool Key, byte[] Data)>();
            await foreach(var item in queue.Reader.ReadAllAsync(cancellationToken))
            {
                packets.Clear();
                using(item.Frame)
                {
                    long timestamp = Math.Max(lastTimestamp + 1, (long)Stopwatch.GetElapsedTime(started).TotalMilliseconds);
                    lastTimestamp = timestamp;
                    if(pending.Count == 16) throw new InvalidOperationException("NVENC is not returning frames in low-latency mode.");
                    pending.Add(timestamp, item.RunId);
                    long before = Stopwatch.GetTimestamp();
                    encoder!.Encode(item.Frame, timestamp, (pts, key, bytes) =>
                    {
                        if(!pending.Remove(pts, out long runId)) throw new InvalidOperationException("NVENC returned an unknown timestamp.");
                        packets.Add((pts, runId, key, bytes));
                    });
                    encodeTicks += Stopwatch.GetTimestamp() - before;
                }
                surfaces.Release();
                foreach(var packet in packets)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(header, packet.Data.Length);
                    header[4] = packet.Key ? (byte)1 : (byte)0;
                    Write(5, packet.Timestamp * 1000);
                    Write(13, packet.RunId);
                    Write(21, Interlocked.Read(ref seen));
                    Write(29, Interlocked.Read(ref skipped));
                    Write(37, ++encoded);
                    Write(45, (long)Stopwatch.GetElapsedTime(0, Interlocked.Read(ref captureTicks)).TotalMicroseconds);
                    Write(53, (long)Stopwatch.GetElapsedTime(0, encodeTicks).TotalMicroseconds);
                    await output.WriteAsync(header, cancellationToken);
                    await output.WriteAsync(packet.Data, cancellationToken);
                    await output.FlushAsync(cancellationToken);
                }
            }
            void Write(int offset, long value) => BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(offset), value);
        }

        public void Dispose()
        {
            lock(captureGate)
            {
                stopped = true;
                queue.Writer.TryComplete();
                while(queue.Reader.TryRead(out var item)) item.Frame.Dispose();
                encoder?.Dispose();
                surfaces.Dispose();
            }
        }
    }
}
