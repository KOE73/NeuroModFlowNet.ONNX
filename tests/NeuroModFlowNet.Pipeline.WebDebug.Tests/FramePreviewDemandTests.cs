using NeuroModFlowNet.ONNX;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class FramePreviewDemandTests
{
    [Fact]
    public async Task AlreadyDownloadedPreviewDoesNotEnterTheConfiguredGpuBackendAgain()
    {
        using var context = Context("camera", 1);
        using OrtValue tensor = OrtValue.CreateTensorValueFromMemory(new byte[800 * 600 * 3], new long[] { 1, 600, 800, 3 });
        context.Set("image.preview.tensor", tensor);
        var inspector = new PausedRegisterInspector(InferenceBackend.TensorRt);
        using CapturedRegister captured = await inspector.CaptureAsync(context, new("image.preview.tensor", 0, "png", 640), TestContext.Current.CancellationToken);
        using Mat image = Cv2.ImDecode(captured.Encode().Data, ImreadModes.Color);
        Assert.Equal(640, image.Width); Assert.Equal(480, image.Height);
    }

    [Fact]
    public void IdleGateDoesNotPrepareOrAllocate()
    {
        using var demand = new FramePreviewDemand();
        demand.ShouldPrepare(false);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for(int index = 0; index < 10000; index++) Assert.False(demand.ShouldPrepare(false));
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public async Task ConcurrentRequestsShareOnePreparedFrameAndNextRequestRequiresANewOne()
    {
        using var demand = new FramePreviewDemand();
        using var image = new Mat(600, 800, MatType.CV_8UC3, new Scalar(10, 30, 90));
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task<byte[]> first = demand.RequestAsync(timeout.Token), second = demand.RequestAsync(timeout.Token);
        Assert.True(demand.ShouldPrepare(false));
        Assert.False(demand.ShouldPrepare(false));
        demand.Publish(image);
        byte[][] results = await Task.WhenAll(first, second);
        Assert.Same(results[0], results[1]);
        using(Mat decoded = Cv2.ImDecode(results[0], ImreadModes.Color)) Assert.Equal(800, decoded.Width);
        Assert.False(demand.ShouldPrepare(false));
        Task<byte[]> next = demand.RequestAsync(timeout.Token);
        Assert.True(demand.ShouldPrepare(false));
        demand.Publish(image);
        Assert.NotSame(results[0], await next);
    }

    [Fact]
    public async Task CancelledAndDisposedRequestsCannotLeavePreparationEnabled()
    {
        using var demand = new FramePreviewDemand();
        using var cancellation = new CancellationTokenSource();
        Task<byte[]> request = demand.RequestAsync(cancellation.Token);
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => request);
        Assert.False(demand.ShouldPrepare(false));
        Task<byte[]> waiting = demand.RequestAsync(CancellationToken.None);
        demand.Dispose();
        await Assert.ThrowsAsync<InvalidOperationException>(() => waiting);
        Assert.False(demand.ShouldPrepare(false));
    }

    [Fact]
    public async Task IdlePreviewCommandsAndWarmupSkipTheirBodiesButRetainTheirProgramPositions()
    {
        using var session = new CameraDebugSession("camera", new(InferenceBackend.Cpu), "camera_warmup",
            new("image.preview.tensor", "image.preview"));
        var resize = new PreviewWrite("image.preview.tensor");
        var copy = new PreviewWrite("image.preview");
        VmProgram program = session.Decorate(new([resize, copy], new Dictionary<string, int>(), "common"));
        session.Command(new("detach"));
        using(var context = Context("camera", 1)) await program.ExecuteAsync(context, TestContext.Current.CancellationToken);
        Assert.Equal(0, resize.Executions); Assert.Equal(0, copy.Executions);
        using var cancellation = new CancellationTokenSource();
        Task<byte[]> request = session.Preview.RequestAsync(cancellation.Token);
        using(var warmup = Context("camera_warmup", 2)) await program.ExecuteAsync(warmup, TestContext.Current.CancellationToken);
        Assert.Equal(0, resize.Executions);
        using(var context = Context("camera", 3)) await program.ExecuteAsync(context, TestContext.Current.CancellationToken);
        Assert.Equal(1, resize.Executions); Assert.Equal(1, copy.Executions);
        Assert.Equal([0, 1], session.Instructions.Select(instruction => instruction.Index));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => request);
        using(var context = Context("camera", 4)) await program.ExecuteAsync(context, TestContext.Current.CancellationToken);
        Assert.Equal(1, resize.Executions); Assert.Equal(1, copy.Executions);
        var command = new DemandPreviewOp(resize, _ => false);
        await command.DisposeAsync();
        Assert.True(resize.Disposed);
    }

    static VmRunContext Context(string source, long run) => new(new(source, run, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());

    sealed class PreviewWrite(string key) : IOp, IDisposable
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create(key, "test", writes: [VarRequirement.Write<int>(key)]);
        public int Executions { get; private set; }
        public bool Disposed { get; private set; }
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
        {
            Executions++; context.Set(key, 1); return ValueTask.FromResult(OpResult.Continue);
        }
        public void Dispose() => Disposed = true;
    }
}
