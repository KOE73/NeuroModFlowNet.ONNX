using System.Diagnostics;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Owns camera capture and VM controller execution for the single-stream lab mode.
/// </summary>
/// <remarks>
/// The host deliberately does not use Spectre.Console. It is responsible only for camera frames, VM inputs, output Mat
/// disposal and optional OpenCV preview windows. Any console presentation is an observer concern.
/// </remarks>
internal sealed class CameraVmHost
{
    readonly VmProgram program;
    readonly CameraVmHostOptions options;

    public CameraVmHost(VmProgram program, CameraVmHostOptions options)
    {
        this.program = program ?? throw new ArgumentNullException(nameof(program));
        this.options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task RunAsync(IVmCameraObserver observer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(observer);

        using var capture = new VideoCapture(options.CameraIndex);
        if(!capture.IsOpened())
            throw new InvalidOperationException($"Camera {options.CameraIndex} is not available.");

        int cameraWidth = checked((int)capture.FrameWidth);
        int cameraHeight = checked((int)capture.FrameHeight);
        observer.CameraOpened(options.CameraIndex, cameraWidth, cameraHeight, capture.Fps);

        var controllerOptions = new VmControllerOptions(
            SourceId: $"camera_{options.CameraIndex}",
            MaxInFlight: options.MaxInFlight,
            OutputKeys: [VmLabKeys.OutputImage],
            RequireWarmup: true,
            CaptureVariablesInTrace: false);

        using var controller = new VmController(controllerOptions, program);

        using Mat warmupFrame = ReadCameraFrame(capture);
        VmRunOutcome warmupOutcome = await controller.WarmupAsync(CreateInputs(warmupFrame), cancellationToken).ConfigureAwait(false);
        DisposeOutputImage(warmupOutcome);
        observer.WarmupCompleted(warmupOutcome);

        if(warmupOutcome.Status != VmRunStatus.Completed)
            throw new InvalidOperationException("Pipeline warmup failed.", warmupOutcome.Exception);

        int frameIndex = 0;
        var runStopwatch = new Stopwatch();
        var loopStopwatch = new Stopwatch();
        TimeSpan cameraFramePeriod = GetCameraFramePeriod(capture.Fps);

        while(!cancellationToken.IsCancellationRequested)
        {
            loopStopwatch.Restart();
            using Mat frame = ReadCameraFrame(capture);
            frameIndex++;

            runStopwatch.Restart();
            VmRunOutcome outcome = await ExecuteOneFrameAsync(controller, frame, frameIndex).ConfigureAwait(false);
            runStopwatch.Stop();

            Mat? outputImage = TryGetOutputImage(outcome);
            observer.FrameCompleted(frameIndex, outcome, runStopwatch.Elapsed);

            bool shouldStop = ShowFrameWindows(frame, outputImage);
            outputImage?.Dispose();

            if(shouldStop)
                break;

            loopStopwatch.Stop();
            await DelayUntilNextCameraFrameAsync(loopStopwatch.Elapsed, cameraFramePeriod, cancellationToken).ConfigureAwait(false);
        }

        Cv2.DestroyAllWindows();
    }

    static VmRunInputs CreateInputs(Mat frame)
    {
        var inputs = new VmRunInputs();

        // The VM reads this Mat through a zero-copy OrtValue wrapper. The host awaits the run before disposing the frame,
        // so the unmanaged OpenCV buffer stays valid for every instruction in this transaction.
        inputs.Add(VmLabKeys.InputImage, frame, disposeWithContext: false);
        return inputs;
    }

    static async Task<VmRunOutcome> ExecuteOneFrameAsync(
        VmController controller,
        Mat frame,
        int frameIndex)
    {
        if(!controller.TryStartRun(CreateInputs(frame), out VmRunHandle handle, sourceFrameId: frameIndex))
            throw new InvalidOperationException("Pipeline controller rejected the frame because MaxInFlight is full.");

        return await handle.Completion.ConfigureAwait(false);
    }

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

    static void DisposeOutputImage(VmRunOutcome outcome)
    {
        Mat? outputImage = TryGetOutputImage(outcome);
        outputImage?.Dispose();
    }

    static Mat? TryGetOutputImage(VmRunOutcome outcome)
    {
        if(outcome.Status == VmRunStatus.Completed &&
            outcome.Output is not null &&
            outcome.Output.TryGet(VmLabKeys.OutputImage, out Mat outputImage))
        {
            return outputImage;
        }

        return null;
    }

    bool ShowFrameWindows(Mat sourceFrame, Mat? outputImage)
    {
        if(!options.ShowOpenCvWindows)
            return false;

        Cv2.ImShow("Camera 0 - source", sourceFrame);

        if(outputImage is not null)
            Cv2.ImShow("Pipeline VM - cropped output", outputImage);

        int key = Cv2.WaitKey(1);
        return key is 27 or 'q' or 'Q';
    }

    TimeSpan GetCameraFramePeriod(double cameraFps)
    {
        if(!options.SyncLoopToCameraFps || cameraFps <= 0)
            return TimeSpan.Zero;

        return TimeSpan.FromSeconds(1.0 / cameraFps);
    }

    static async ValueTask DelayUntilNextCameraFrameAsync(
        TimeSpan elapsed,
        TimeSpan cameraFramePeriod,
        CancellationToken cancellationToken)
    {
        if(cameraFramePeriod <= TimeSpan.Zero || elapsed >= cameraFramePeriod)
            return;

        // Camera drivers often return the last captured buffer immediately when polled faster than the physical FPS.
        // The delay keeps the visual loop near camera cadence while pipeline timings still measure only VM execution.
        await Task.Delay(cameraFramePeriod - elapsed, cancellationToken).ConfigureAwait(false);
    }
}
