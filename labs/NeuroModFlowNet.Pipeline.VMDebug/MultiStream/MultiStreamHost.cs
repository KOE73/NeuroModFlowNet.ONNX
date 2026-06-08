using System.Diagnostics;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Orchestrates the multi-stream lab: opens the camera, creates VM workers and the OBB batch service,
/// fans out each camera frame to all VM workers, and drives the whole pipeline until cancellation.
/// </summary>
/// <remarks>
/// Stream emulation:
///   There is one physical camera. Each frame is cloned once per VM worker and written into that worker's bounded
///   channel. Workers are independent Tasks that run their own <see cref="VmController"/> concurrently.
///   This emulates N independent camera streams feeding separate VM instances without requiring N real cameras.
///
/// OBB batch service:
///   Each worker also submits a copy of its raw camera frame to <see cref="ObbBatchService"/>. The service
///   accumulates frames from all workers, concatenates them into a single batch via a generated ONNX Concat graph,
///   and runs the OBB model once per accumulated batch.
/// </remarks>
internal sealed class MultiStreamHost
{
    readonly VmProgram program;
    readonly MultiStreamOptions options;
    readonly string obbModelPath;

    public IReadOnlyList<VmWorker> Workers { get; private set; } = [];

    public ObbBatchService? ObbService { get; private set; }

    public MultiStreamHost(VmProgram program, MultiStreamOptions options, string obbModelPath)
    {
        this.program = program ?? throw new ArgumentNullException(nameof(program));
        this.options = options ?? throw new ArgumentNullException(nameof(options));
        ArgumentException.ThrowIfNullOrWhiteSpace(obbModelPath);
        this.obbModelPath = obbModelPath;
    }

    public async Task RunAsync(IMultiStreamObserver observer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(observer);

        using var capture = new VideoCapture(options.CameraIndex);
        if(!capture.IsOpened())
            throw new InvalidOperationException($"Camera {options.CameraIndex} is not available.");

        int cameraWidth = checked((int)capture.FrameWidth);
        int cameraHeight = checked((int)capture.FrameHeight);
        observer.CameraOpened(options.CameraIndex, cameraWidth, cameraHeight, capture.Fps);

        using var obbService = new ObbBatchService(obbModelPath, options.ObbBatchSize);
        ObbService = obbService;

        var workers = new List<VmWorker>(options.VmCount);
        for(int vmIndex = 0; vmIndex < options.VmCount; vmIndex++)
            workers.Add(new VmWorker(vmIndex, program, obbService, options));

        Workers = workers;

        try
        {
            // Start OBB service background loop.
            Task obbTask = obbService.RunAsync(cancellationToken);

            // Start all VM workers.
            List<Task> workerTasks = workers.Select(worker => worker.RunAsync(cancellationToken)).ToList();

            // Send the first warmup frame to every worker (they each need one frame to proceed past warmup).
            BroadcastFirstFrame(capture, workers);

            observer.AllWorkersWarmedUp();

            // Camera fan-out loop.
            TimeSpan framePeriod = GetCameraFramePeriod(capture.Fps);
            var loopStopwatch = new Stopwatch();

            while(!cancellationToken.IsCancellationRequested)
            {
                loopStopwatch.Restart();

                using Mat frame = ReadCameraFrame(capture);
                BroadcastFrame(frame, workers);

                loopStopwatch.Stop();
                await DelayUntilNextFrameAsync(loopStopwatch.Elapsed, framePeriod, cancellationToken).ConfigureAwait(false);
            }

            // Complete all worker channels so ReadAllAsync terminates.
            foreach(VmWorker worker in workers)
                worker.FrameInput.TryComplete();

            await Task.WhenAll(workerTasks).ConfigureAwait(false);
            await obbTask.ConfigureAwait(false);
        }
        finally
        {
            foreach(VmWorker worker in workers)
                worker.Dispose();

            ObbService = null;

            if(options.ShowOpenCvWindows)
                Cv2.DestroyAllWindows();
        }
    }

    #region Camera helpers

    static Mat ReadCameraFrame(VideoCapture capture)
    {
        var frame = new Mat();

        if(!capture.Read(frame) || frame.Empty())
        {
            frame.Dispose();
            throw new InvalidOperationException("Camera returned an empty frame.");
        }

        return frame;
    }

    static void BroadcastFirstFrame(VideoCapture capture, List<VmWorker> workers)
    {
        using Mat firstFrame = ReadCameraFrame(capture);

        // Each worker needs its own clone for the warmup step.
        foreach(VmWorker worker in workers)
        {
            Mat clone = firstFrame.Clone();
            if(!worker.FrameInput.TryWrite(clone))
                clone.Dispose();
        }
    }

    static void BroadcastFrame(Mat frame, List<VmWorker> workers)
    {
        // Clone the frame for every worker. The worker owns the clone and disposes it after its run.
        foreach(VmWorker worker in workers)
        {
            Mat clone = frame.Clone();
            if(!worker.FrameInput.TryWrite(clone))
                clone.Dispose(); // Channel full (DropOldest) already handled internally, but TryWrite may still return false.
        }
    }

    TimeSpan GetCameraFramePeriod(double cameraFps)
    {
        if(!options.SyncLoopToCameraFps || cameraFps <= 0)
            return TimeSpan.Zero;

        return TimeSpan.FromSeconds(1.0 / cameraFps);
    }

    static async ValueTask DelayUntilNextFrameAsync(
        TimeSpan elapsed,
        TimeSpan framePeriod,
        CancellationToken cancellationToken)
    {
        if(framePeriod <= TimeSpan.Zero || elapsed >= framePeriod)
            return;

        await Task.Delay(framePeriod - elapsed, cancellationToken).ConfigureAwait(false);
    }

    #endregion
}
