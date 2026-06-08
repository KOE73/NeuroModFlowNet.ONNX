using System.Diagnostics;
using System.Threading.Channels;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// One emulated camera stream: owns a <see cref="VmController"/>, reads frames from its channel,
/// runs the VM program, shows an OpenCV preview window, and submits a copy of the raw frame to the OBB batch service.
/// </summary>
/// <remarks>
/// Each VM worker is an independent async Task. The camera host broadcasts cloned Mats into each worker's channel.
/// Workers run fully concurrently — they share the same <see cref="VmProgram"/> definition but each has its
/// own <see cref="VmController"/> instance, so no VM state is shared.
/// </remarks>
internal sealed class VmWorker : IDisposable
{
    readonly VmProgram program;
    readonly ObbBatchService obbService;
    readonly MultiStreamOptions options;
    readonly Channel<Mat> frameChannel;
    readonly VmController controller;

    public int VmIndex { get; }

    public VmSlotStatistics Statistics { get; }

    /// <summary>Writer side exposed to the camera host for frame fan-out.</summary>
    public ChannelWriter<Mat> FrameInput => frameChannel.Writer;

    public VmWorker(
        int vmIndex,
        VmProgram program,
        ObbBatchService obbService,
        MultiStreamOptions options)
    {
        VmIndex = vmIndex;
        Statistics = new VmSlotStatistics(vmIndex);

        this.program = program ?? throw new ArgumentNullException(nameof(program));
        this.obbService = obbService ?? throw new ArgumentNullException(nameof(obbService));
        this.options = options ?? throw new ArgumentNullException(nameof(options));

        // Bounded channel: if this VM falls behind the camera, old frames are dropped instead of accumulating unboundedly.
        frameChannel = Channel.CreateBounded<Mat>(new BoundedChannelOptions(capacity: 4)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true
        });

        var controllerOptions = new VmControllerOptions(
            SourceId: $"vm_{vmIndex}",
            MaxInFlight: options.MaxInFlightPerVm,
            OutputKeys: [VmLabKeys.OutputImage],
            RequireWarmup: true,
            CaptureVariablesInTrace: false);

        controller = new VmController(controllerOptions, program);
    }

    /// <summary>
    /// Warms up the VM controller, then enters the frame processing loop until cancellation.
    /// </summary>
    public async Task RunAsync(CancellationToken cancellationToken)
    {
        // Warmup: read first available frame from the channel.
        Mat warmupFrame = await frameChannel.Reader.ReadAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            VmRunOutcome warmupOutcome = await controller
                .WarmupAsync(CreateInputs(warmupFrame), cancellationToken)
                .ConfigureAwait(false);

            DisposeOutputImage(warmupOutcome);
            Statistics.AddWarmup(warmupOutcome);

            if(warmupOutcome.Status != VmRunStatus.Completed)
                throw new InvalidOperationException($"VM {VmIndex} warmup failed.", warmupOutcome.Exception);
        }
        finally
        {
            warmupFrame.Dispose();
        }

        int frameIndex = 0;
        var runStopwatch = new Stopwatch();

        await foreach(Mat frame in frameChannel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            frameIndex++;

            // Clone the raw frame before VM execution so the OBB service gets the original camera image,
            // not the cropped/resized VM output.
            Mat frameForObb = frame.Clone();

            try
            {
                runStopwatch.Restart();
                VmRunOutcome outcome = await ExecuteOneFrameAsync(frame, frameIndex, cancellationToken).ConfigureAwait(false);
                runStopwatch.Stop();

                Statistics.AddFrame(outcome, runStopwatch.Elapsed);

                Mat? outputImage = TryGetOutputImage(outcome);

                if(options.ShowOpenCvWindows)
                    ShowVmWindow(frame, outputImage);

                outputImage?.Dispose();

                // Submit the original camera frame to the OBB service after VM run completes.
                // Ownership of frameForObb transfers to the service.
                obbService.SubmitFrame(frameForObb);
                frameForObb = null!; // prevent double-dispose in finally
            }
            finally
            {
                frameForObb?.Dispose();
                frame.Dispose();
            }
        }

        if(options.ShowOpenCvWindows)
            Cv2.DestroyWindow(WindowName);
    }

    #region Private helpers

    string WindowName => $"VM {VmIndex} — output";

    static VmRunInputs CreateInputs(Mat frame)
    {
        var inputs = new VmRunInputs();
        inputs.Add(VmLabKeys.InputImage, frame, disposeWithContext: false);
        return inputs;
    }

    async Task<VmRunOutcome> ExecuteOneFrameAsync(
        Mat frame,
        int frameIndex,
        CancellationToken cancellationToken)
    {
        if(!controller.TryStartRun(CreateInputs(frame), out VmRunHandle handle, sourceFrameId: frameIndex))
            throw new InvalidOperationException($"VM {VmIndex} controller rejected frame — MaxInFlight full.");

        return await handle.Completion.ConfigureAwait(false);
    }

    static void DisposeOutputImage(VmRunOutcome outcome)
    {
        TryGetOutputImage(outcome)?.Dispose();
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

    void ShowVmWindow(Mat sourceFrame, Mat? outputImage)
    {
        if(outputImage is not null)
            Cv2.ImShow(WindowName, outputImage);
        else
            Cv2.ImShow(WindowName, sourceFrame);

        Cv2.WaitKey(1);
    }

    #endregion

    public void Dispose()
    {
        controller.Dispose();
        frameChannel.Writer.TryComplete();
    }
}
