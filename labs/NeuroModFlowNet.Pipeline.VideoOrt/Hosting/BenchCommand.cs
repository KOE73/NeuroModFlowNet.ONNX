using NeuroModFlowNet.Pipeline.Video.Nvdec;
using System.Diagnostics;
using NeuroModFlowNet.Pipeline.ONNX;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.VideoOrt;

/// <summary>
/// EN: Throughput of N independent camera streams, each with its own NVDEC decoder thread and its own
/// <see cref="VmController"/> running [NV12→BGR, resize] fully on the GPU (no host copy). Files loop. With
/// <c>--decode-only</c> the VM is skipped to measure NVDEC alone.
///
/// RU: Пропускная способность N независимых потоков камер: у каждого свой поток декодера NVDEC и свой
/// <see cref="VmController"/>, выполняющий [NV12→BGR, resize] полностью на GPU (без копии в host). Файлы зациклены.
/// С <c>--decode-only</c> VM пропускается, чтобы измерить только NVDEC.
/// </summary>
internal static class BenchCommand
{
    public static async Task<int> RunAsync(VideoOrtOptions options, CancellationToken cancellationToken)
    {
        var streams = new StreamStatistics[options.Streams];
        for(int index = 0; index < streams.Length; index++)
            streams[index] = new StreamStatistics(options.Sources[index % options.Sources.Count]);

        using var stopSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        stopSource.CancelAfter(TimeSpan.FromSeconds(options.Seconds));

        Console.WriteLine($"bench: {options.Streams} streams, {options.Seconds} s, backend {options.Backend}, fence {(options.UseFence ? "on" : "off")}, sessions {options.SessionSharing}, " +
                          $"{(options.DecodeOnly ? "decode only" : $"VM NV12->BGR + resize {options.ResizeWidth}x{options.ResizeHeight}, max in flight {options.MaxInFlight}")}");

        Task[] tasks = streams
            .Select((statistics, index) => Task.Factory.StartNew(
                () => RunStreamAsync(index, statistics, options, stopSource.Token),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default).Unwrap())
            .ToArray();

        var clock = Stopwatch.StartNew();
        long lastTotal = 0;
        TimeSpan lastTime = TimeSpan.Zero;
        while(!Task.WaitAll(tasks, TimeSpan.FromSeconds(2)))
        {
            long total = streams.Sum(static stream => stream.Completed);
            TimeSpan now = clock.Elapsed;
            double fps = (total - lastTotal) / (now - lastTime).TotalSeconds;
            Console.WriteLine($"{now.TotalSeconds,5:F0}s total {fps,7:F1} fps  per stream {fps / streams.Length,6:F1}  " +
                              $"vm {streams.Average(static stream => stream.AverageRunMilliseconds),5:F2} ms  " +
                              $"decode {streams.Average(static stream => stream.AverageDecodeMilliseconds),5:F2} ms");
            lastTotal = total;
            lastTime = now;
        }

        double seconds = clock.Elapsed.TotalSeconds;
        Console.WriteLine("stream  source                                   frames     fps   vm ms  decode ms  error");
        for(int index = 0; index < streams.Length; index++)
        {
            StreamStatistics stream = streams[index];
            Console.WriteLine($"{index,6}  {Path.GetFileName(stream.Source),-40} {stream.Completed,6} {stream.Completed / seconds,7:F1} {stream.AverageRunMilliseconds,7:F2} {stream.AverageDecodeMilliseconds,10:F2}  {stream.Error}");
        }

        long all = streams.Sum(static stream => stream.Completed);
        Console.WriteLine($"TOTAL {all} frames in {seconds:F1} s = {all / seconds:F1} fps ({all / seconds / streams.Length:F1} per stream)");
        return streams.Any(static stream => stream.Error is not null) ? 1 : 0;
    }

    static async Task RunStreamAsync(int index, StreamStatistics statistics, VideoOrtOptions options, CancellationToken cancellationToken)
    {
        try
        {
            using var decoder = new NvdecVideoDecoder(statistics.Source, loop: true, useFence: options.UseFence, extraHardwareFrames: options.MaxInFlight + 4);
            if(options.DecodeOnly)
            {
                await DecodeOnlyAsync(decoder, statistics, cancellationToken).ConfigureAwait(false);
                return;
            }

            VmProgram program = new VmProgramBuilder($"video.stream{index}")
                .Step(new Op_Onnx_Nv12_To_BgrU8Nhwc(VideoOrtKeys.Nv12, VideoOrtKeys.Bgr, decoder.Width, decoder.Height, options.Matrix, isFinal: false, executionBackend: options.Backend))
                .Step(new Op_Onnx_Resize_U8_NHWC(VideoOrtKeys.Bgr, VideoOrtKeys.Resized, new CvSize(options.ResizeWidth, options.ResizeHeight), isFinal: false, executionBackend: options.Backend))
                .Build();

            await using var controller = new VmController(
                new VmControllerOptions($"stream{index}", MaxInFlight: options.MaxInFlight, RequireWarmup: true, CaptureVariablesInTrace: false),
                program);

            NvdecFrame warmupFrame = decoder.ReadNext() ?? throw new InvalidOperationException("Empty source.");
            VmRunOutcome warmup = await controller.WarmupAsync(CreateInputs(warmupFrame), cancellationToken).ConfigureAwait(false);
            if(warmup.Status != VmRunStatus.Completed)
                throw new InvalidOperationException("Warmup failed.", warmup.Exception);

            var pending = new Queue<Task>();
            var decodeClock = new Stopwatch();
            while(!cancellationToken.IsCancellationRequested)
            {
                decodeClock.Restart();
                NvdecFrame frame = decoder.ReadNext() ?? throw new InvalidOperationException("Looping source ended.");
                statistics.AddDecode(decodeClock.Elapsed);

                if(pending.Count >= options.MaxInFlight)
                    await pending.Dequeue().ConfigureAwait(false);

                if(!controller.TryStartRun(CreateInputs(frame), out VmRunHandle handle, sourceFrameId: frame.FrameIndex))
                    throw new InvalidOperationException("Controller rejected a run despite the in-flight limit.");

                pending.Enqueue(ObserveAsync(handle, statistics));
            }

            await Task.WhenAll(pending).ConfigureAwait(false);
        }
        catch(OperationCanceledException) when(cancellationToken.IsCancellationRequested)
        {
        }
        catch(Exception exception)
        {
            statistics.Error = exception.GetBaseException().Message;
        }
    }

    static Task DecodeOnlyAsync(NvdecVideoDecoder decoder, StreamStatistics statistics, CancellationToken cancellationToken)
    {
        var decodeClock = new Stopwatch();
        while(!cancellationToken.IsCancellationRequested)
        {
            decodeClock.Restart();
            using NvdecFrame frame = decoder.ReadNext() ?? throw new InvalidOperationException("Looping source ended.");
            statistics.AddDecode(decodeClock.Elapsed);
            statistics.AddRun(TimeSpan.Zero);
        }

        return Task.CompletedTask;
    }

    static async Task ObserveAsync(VmRunHandle handle, StreamStatistics statistics)
    {
        var clock = Stopwatch.StartNew();
        VmRunOutcome outcome = await handle.Completion.ConfigureAwait(false);
        if(outcome.Status != VmRunStatus.Completed)
            throw new InvalidOperationException("Run failed.", outcome.Exception);

        statistics.AddRun(clock.Elapsed);
    }

    static VmRunInputs CreateInputs(NvdecFrame frame) =>
        new VmRunInputs()
            .Add(VideoOrtKeys.Frame, frame, disposeWithContext: true)
            .Add(VideoOrtKeys.Nv12, frame.CreateNv12Tensor(), disposeWithContext: true);

    sealed class StreamStatistics(string source)
    {
        readonly object syncRoot = new();
        long completed;
        double runMilliseconds;
        double decodeMilliseconds;
        long decodes;

        public string Source { get; } = source;

        public string? Error { get; set; }

        public long Completed => Interlocked.Read(ref completed);

        public double AverageRunMilliseconds
        {
            get { lock(syncRoot) return completed == 0 ? 0 : runMilliseconds / completed; }
        }

        public double AverageDecodeMilliseconds
        {
            get { lock(syncRoot) return decodes == 0 ? 0 : decodeMilliseconds / decodes; }
        }

        public void AddRun(TimeSpan elapsed)
        {
            lock(syncRoot)
            {
                runMilliseconds += elapsed.TotalMilliseconds;
                completed++;
            }
        }

        public void AddDecode(TimeSpan elapsed)
        {
            lock(syncRoot)
            {
                decodeMilliseconds += elapsed.TotalMilliseconds;
                decodes++;
            }
        }
    }
}
