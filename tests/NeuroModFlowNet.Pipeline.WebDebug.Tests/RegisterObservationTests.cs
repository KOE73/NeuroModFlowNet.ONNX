using System.Text;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class RegisterObservationTests
{
    #region Demand, backpressure and lifecycle

    [Fact]
    public async Task PreviewProducerChecksLeaseAcknowledgementAndFrequencyBeforeItsDownload()
    {
        using var observation = new RegisterObservation((context, request, cancellation) => ValueTask.FromResult(Value("7")));
        using var context = Context();
        var target = new DebugWatchTarget("image.preview.tensor", "json", "common", 13, "after");
        var configuration = new DebugWatchConfiguration(true, 0.2, 640, [new(target.Key, target.Format, target.Program, target.Index, target.Phase)]);
        DebugWatchSnapshot snapshot = observation.Configure(configuration, [target]);
        Assert.False(observation.NeedsPreviewCapture("common", 13));
        observation.Poll(new(snapshot.Generation, []));
        Assert.True(observation.NeedsPreviewCapture("common", 13));
        Assert.False(observation.NeedsPreviewCapture("common", 14));
        await observation.CaptureAsync(context, "common", 13, "after", "preview", TestContext.Current.CancellationToken);
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 1);
        Assert.False(observation.NeedsPreviewCapture("common", 13));
        long sequence = observation.Snapshot().Items[0].Frame!.Sequence;
        observation.Poll(new(snapshot.Generation, new() { [target.Key] = sequence }));
        Assert.False(observation.NeedsPreviewCapture("common", 13));
        observation.Configure(configuration with { Enabled = false }, [target]);
        Assert.False(observation.NeedsPreviewCapture("common", 13));
    }

    [Fact]
    public async Task DisabledObservationDoesNotReadRegistersOrAllocatePerVisit()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context, request, cancellation) =>
        {
            captures++;
            throw new InvalidOperationException("Disabled observation must not reach preparation.");
        });
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(new(false,3,640,[new("value","json")]),[Target]);
        observation.Poll(new(snapshot.Generation,[]));
        await observation.CaptureAsync(context,"common",0,"after","write",CancellationToken.None);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for(int index = 0; index < 1000; index++)
            observation.CaptureAsync(context,"common",0,"after","write",CancellationToken.None).GetAwaiter().GetResult();
        Assert.Equal(0,GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(0,captures);
    }

    [Fact]
    public async Task BusyCaptureAndUnacknowledgedDeliveryAreSkippedBeforePreparation()
    {
        var prepared = new TaskCompletionSource<CapturedRegister>(TaskCreationOptions.RunContinuationsAsynchronously);
        int captures = 0;
        using var observation = new RegisterObservation((context, request, cancellation) => { captures++; return new(prepared.Task); });
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(Configuration(),[Target]);
        await Capture(observation,context);
        Assert.Equal(0,captures); // No browser demand, even though registers were selected.
        observation.Poll(new(snapshot.Generation,[]));
        Task pending = Capture(observation,context);
        Assert.Equal(1,captures);
        await Capture(observation,context);
        prepared.SetResult(Value("41"));
        await pending;
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 1);
        for(int index = 0; index < 20; index++) await Capture(observation,context);
        Assert.Equal(1,captures);
        DebugWatchSnapshot delivered = observation.Poll(new(snapshot.Generation,[]));
        Assert.Equal("41",Encoding.UTF8.GetString(delivered.Items[0].Frame!.Data!));
        await Capture(observation,context);
        Assert.Equal(1,captures); // Sending is not an acknowledgement of browser decoding/display.
        Assert.True(observation.Snapshot().Items[0].Skipped >= 22);
    }

    [Fact]
    public async Task AcknowledgementAllowsNextSampleButFrequencyStillLimitsPreparation()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context, request, cancellation) => { captures++; return ValueTask.FromResult(Value("7")); });
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(Configuration() with { FramesPerSecond = 0.2 },[Target]);
        observation.Poll(new(snapshot.Generation,[]));
        await Capture(observation,context);
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 1);
        long sequence = observation.Snapshot().Items[0].Frame!.Sequence;
        observation.Poll(new(snapshot.Generation,new() { ["value"] = sequence }));
        await Capture(observation,context);
        Assert.Equal(1,captures);
        Assert.Equal(1,observation.Snapshot().Items[0].Skipped);
    }

    [Fact]
    public async Task ReconfigurationDiscardsAnInFlightOldSample()
    {
        var prepared = new TaskCompletionSource<CapturedRegister>(TaskCreationOptions.RunContinuationsAsynchronously);
        using var observation = new RegisterObservation((context, request, cancellation) => new(prepared.Task));
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(Configuration(),[Target]);
        observation.Poll(new(snapshot.Generation,[]));
        Task pending = Capture(observation,context);
        DebugWatchSnapshot disabled = observation.Configure(Configuration() with { Enabled = false },[Target]);
        prepared.SetResult(Value("old"));
        await pending;
        Assert.False(observation.Enabled);
        Assert.Equal(disabled.Generation,observation.Snapshot().Generation);
        Assert.Null(observation.Snapshot().Items[0].Frame);
        Assert.Equal(0,observation.Snapshot().Items[0].Captured);
    }

    [Fact]
    public async Task ObservationFailureIsReportedWithoutFailingTheVmOrTryingAnotherFormat()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context, request, cancellation) =>
        {
            captures++;
            throw new NotSupportedException("Chosen provider path is unavailable.");
        });
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(Configuration(),[Target]);
        observation.Poll(new(snapshot.Generation,[]));
        await Capture(observation,context);
        DebugWatchFrame frame = observation.Poll(new(snapshot.Generation,[])).Items[0].Frame!;
        Assert.Contains("provider path",frame.Error);
        Assert.Null(frame.Data);
        Assert.Equal(1,captures);
    }

    [Fact]
    public async Task JsonCaptureRemainsValidAfterTheRunContextAndSourceArrayAreGone()
    {
        var inspector = new PausedRegisterInspector(InferenceBackend.Cpu,Nv12ColorMatrix.Bt601Full);
        int[] values = [2,3,5];
        CapturedRegister captured;
        using(var context = Context())
        {
            context.Set("value",values);
            captured = await inspector.CaptureAsync(context,new("value",0),CancellationToken.None);
            values[0] = 999;
        }
        using(captured)
        using(var json = System.Text.Json.JsonDocument.Parse(captured.Encode().Data))
            Assert.Equal(new[] {2,3,5},json.RootElement.EnumerateArray().Select(item => item.GetInt32()));
    }

    [Fact]
    public async Task SessionObservesSourceBeforeFirstInstructionAndOutputAfterItsWriterAcrossRuns()
    {
        using var session = new CameraDebugSession("camera",new PausedRegisterInspector(InferenceBackend.Cpu,Nv12ColorMatrix.Bt601Full));
        VmProgram program = session.Decorate(new([new WriteOp()],new Dictionary<string,int>(),"common"));
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var firstContext = Context(); firstContext.Set("source",41);
        Task firstRun = program.ExecuteAsync(firstContext,timeout.Token).AsTask();
        await WaitFor(() => session.Snapshot().State == "paused");
        session.Command(new("step"));
        await WaitFor(() => session.Snapshot().Location?.Phase == "after");
        DebugWatchSnapshot configured = session.ConfigureObservation(new(true,15,640,[new("source","json"),new("value","json")]));
        Assert.Equal("before",configured.Items[0].Target.Phase);
        Assert.Equal("after",configured.Items[1].Target.Phase);
        session.Observation.Poll(new(configured.Generation,[]));
        session.Command(new("continue"));
        await firstRun;
        using var secondContext = new VmRunContext(new("camera",2,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());
        secondContext.Set("source",42);
        await program.ExecuteAsync(secondContext,timeout.Token);
        await WaitFor(() => session.Observation.Snapshot().Items.All(item => item.Captured == 1));
        DebugWatchSnapshot frames = session.Observation.Poll(new(configured.Generation,[]));
        Assert.All(frames.Items,item => Assert.Equal(2,item.Frame!.Location.RunId));
        Assert.Equal("42",Encoding.UTF8.GetString(frames.Items[0].Frame!.Data!));
        Assert.Equal("43",Encoding.UTF8.GetString(frames.Items[1].Frame!.Data!));
        Assert.Equal("running",session.Snapshot().State);
    }

    [Fact]
    public async Task BrowserLeaseExpiryStopsFurtherPreparation()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context,request,cancellation) => { captures++; return ValueTask.FromResult(Value("1")); });
        using var context = Context();
        DebugWatchSnapshot snapshot = observation.Configure(Configuration(),[Target]);
        observation.Poll(new(snapshot.Generation,[]));
        await Task.Delay(TimeSpan.FromSeconds(5.1));
        await Capture(observation,context);
        Assert.False(observation.Enabled);
        Assert.Equal(0,captures);
    }

    [Theory]
    [InlineData(0,640)]
    [InlineData(16,640)]
    [InlineData(3,99)]
    [InlineData(3,8193)]
    public void InvalidLimitsAreRejected(double fps, int maxEdge)
    {
        using var observation = new RegisterObservation((context,request,cancellation) => throw new InvalidOperationException());
        Assert.Throws<ArgumentException>(() => observation.Configure(Configuration() with { FramesPerSecond = fps, MaxEdge = maxEdge },[Target]));
    }

    [Fact]
    public async Task IdenticalViewerTargetsShareOnePreparationAndEncodedPayload()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context,request,cancellation) => { captures++; return ValueTask.FromResult(Value("shared")); });
        using var context = Context();
        DebugWatchSnapshot first = observation.Configure(Configuration(),[Target],"main");
        DebugWatchSnapshot second = observation.Configure(Configuration(),[Target],"screen2");
        observation.Poll(new(first.Generation,[]),"main"); observation.Poll(new(second.Generation,[]),"screen2");
        await Capture(observation,context);
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 1);
        DebugWatchFrame firstFrame = observation.Poll(new(first.Generation,[]),"main").Items[0].Frame!;
        DebugWatchFrame secondFrame = observation.Poll(new(second.Generation,[]),"screen2").Items[0].Frame!;
        Assert.Equal(1,captures); Assert.Same(firstFrame.Data,secondFrame.Data);
        Assert.Equal(firstFrame.Sequence,secondFrame.Sequence);
    }

    [Fact]
    public async Task DisablingMainViewerDoesNotDisableAutonomousViewer()
    {
        using var observation = new RegisterObservation((context,request,cancellation) => ValueTask.FromResult(Value("independent")));
        using var context = Context();
        observation.Configure(Configuration(),[Target]);
        DebugWatchSnapshot second = observation.Configure(Configuration(),[Target],"screen2");
        observation.Poll(new(second.Generation,[]),"screen2");
        observation.Configure(Configuration() with { Enabled = false },[Target]);
        Assert.True(observation.Enabled);
        await Capture(observation,context);
        await WaitFor(() => observation.Snapshot(false,"screen2").Items[0].Captured == 1);
        Assert.Equal(0,observation.Snapshot().Items[0].Captured);
    }

    [Fact]
    public async Task SlowViewerKeepsItsPendingSampleWhileFastViewerContinues()
    {
        int captures = 0;
        using var observation = new RegisterObservation((context,request,cancellation) => ValueTask.FromResult(Value((++captures).ToString())));
        using var context = Context();
        DebugWatchSnapshot first = observation.Configure(Configuration(),[Target]);
        DebugWatchSnapshot second = observation.Configure(Configuration(),[Target],"slow");
        observation.Poll(new(first.Generation,[])); observation.Poll(new(second.Generation,[]),"slow");
        await Capture(observation,context);
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 1);
        long sequence = observation.Snapshot().Items[0].Frame!.Sequence;
        observation.Poll(new(first.Generation,new() { ["value"] = sequence }));
        await Task.Delay(100); await Capture(observation,context);
        await WaitFor(() => observation.Snapshot().Items[0].Captured == 2);
        Assert.Equal(1,observation.Snapshot(false,"slow").Items[0].Captured);
        Assert.Equal(sequence,observation.Snapshot(false,"slow").Items[0].Frame!.Sequence);
        Assert.Equal(2,captures);
    }

    #endregion

    #region Fixtures

    sealed class WriteOp : IOp
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create("write","write",[VarRequirement.Read<int>("source")],[VarRequirement.Write<int>("value")]);
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context,CancellationToken cancellationToken)
        {
            context.Set("value",context.Get<int>("source") + 1);
            return ValueTask.FromResult(OpResult.Continue);
        }
    }

    static DebugWatchTarget Target => new("value","json","common",0,"after");
    static DebugWatchConfiguration Configuration() => new(true,15,640,[new("value","json")]);
    static VmRunContext Context() => new(new("camera",1,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());
    static CapturedRegister Value(string text) => new(null,new("application/json",Encoding.UTF8.GetBytes(text)));
    static Task Capture(RegisterObservation observation,VmRunContext context) => observation.CaptureAsync(context,"common",0,"after","write",CancellationToken.None).AsTask();
    static async Task WaitFor(Func<bool> predicate)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while(!predicate()) await Task.Delay(5,timeout.Token);
    }

    #endregion
}
