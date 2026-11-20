using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class CameraDebugSessionTests
{
    #region Execution and ownership
    [Fact]
    public async Task StepStopsAfterExactlyOneInstructionAndKeepsRegistersAlive()
    {
        using var session = CreateSession();
        var first = new CounterOp("first");
        var second = new CounterOp("second");
        VmProgram program = session.Decorate(new([first, second], new Dictionary<string, int>(), "prepare"));
        using var context = Context(1);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        Task execution = program.ExecuteAsync(context, timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        Assert.Equal(0, first.Count);
        session.Command(new("step", session.Snapshot().Revision));
        await WaitForAsync(() => session.Snapshot().Location?.Phase == "after");
        Assert.Equal(1, first.Count);
        Assert.Equal(0, second.Count);
        Assert.Equal(1, context.Get<int>("first"));
        DebugSnapshot snapshot = session.Snapshot();
        DebugInspectionResult inspection = await session.InspectAsync(new("first", snapshot.Revision), timeout.Token);
        Assert.Equal("1", System.Text.Encoding.UTF8.GetString(inspection.Data));
        session.Command(new("step", session.Snapshot().Revision));
        await WaitForAsync(() => session.Snapshot().Location is { Index: 1, Phase: "after" });
        Assert.Equal(1, second.Count);
        session.Command(new("continue", session.Snapshot().Revision));
        await execution;
    }

    [Fact]
    public async Task BreakpointsAreProgramLocalAndNextFrameUsesRunIdentity()
    {
        using var session = CreateSession();
        VmProgram prepare = session.Decorate(new([new CounterOp("prepare")], new Dictionary<string,int>(), "prepare"));
        VmProgram common = session.Decorate(new([new CounterOp("common")], new Dictionary<string,int>(), "common"));
        session.SetBreakpoints([new("common", 0)]);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var context = Context(1);
        Task run = RunChain();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        session.Command(new("continue"));
        await WaitForAsync(() => session.Snapshot().Location?.Program == "common");
        Assert.Equal("before", session.Snapshot().Location!.Phase);
        session.SetBreakpoints([]);
        session.Command(new("next-frame"));
        await run;
        using var nextContext = Context(2);
        Task nextRun = prepare.ExecuteAsync(nextContext, timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().Location?.RunId == 2);
        Assert.False(nextContext.Contains("prepare"));
        session.Command(new("detach"));
        await nextRun;

        async Task RunChain()
        {
            await prepare.ExecuteAsync(context, timeout.Token);
            await common.ExecuteAsync(context, timeout.Token);
        }
    }

    [Fact]
    public async Task CancellationReleasesPausedContextAndOldInspectionIsRejected()
    {
        using var session = CreateSession();
        VmProgram program = session.Decorate(new([new CounterOp("value")], new Dictionary<string,int>(), "prepare"));
        using var context = Context(1);
        using var cancellation = new CancellationTokenSource();
        Task execution = program.ExecuteAsync(context, cancellation.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        long revision = session.Snapshot().Revision;
        Assert.Throws<InvalidOperationException>(() => { _ = session.InspectAsync(new("value", revision + 1), CancellationToken.None); });
        Assert.Throws<InvalidOperationException>(() => session.Command(new("step", revision + 1)));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => execution);
        Assert.Empty(session.Snapshot().Variables);
    }

    [Fact]
    public async Task DecoratorPreservesValidationAndDisposesNativeInstructions()
    {
        using var session = CreateSession();
        var plain = new CounterOp("plain");
        var device = new DeviceOp();
        VmProgram program = session.Decorate(new([plain, device], new Dictionary<string,int>(), "program"));
        Assert.False(program.Instructions[0] is IHasExecutionDevice);
        Assert.True(((IHasExecutionDevice)program.Instructions[1]).IsGpuExecution);
        Assert.Same(device.Descriptor, program.Instructions[1].Descriptor);
        await ((IAsyncDisposable)program.Instructions[1]).DisposeAsync();
        Assert.True(device.Disposed);
    }

    [Fact]
    public async Task WarmupDoesNotPauseAndDetachedRunsDoNotCaptureRegisters()
    {
        using var session = CreateSession();
        var instruction = new CounterOp("value");
        VmProgram program = session.Decorate(new([instruction], new Dictionary<string,int>(), "program"));
        using var warmup = new VmRunContext(new("camera_warmup", 1, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());
        await program.ExecuteAsync(warmup, CancellationToken.None);
        Assert.Equal(1, instruction.Count);
        session.Command(new("detach"));
        using var context = Context(2);
        await program.ExecuteAsync(context, CancellationToken.None);
        Assert.Equal(2, instruction.Count);
        Assert.Empty(session.Snapshot().Variables);
    }

    [Fact]
    public async Task PausingOneCameraDoesNotStopAnotherController()
    {
        using var firstSession = CreateSession();
        using var secondSession = new CameraDebugSession("second", new PausedRegisterInspector(InferenceBackend.Cpu));
        var firstOperation = new CounterOp("first");
        var secondOperation = new CounterOp("second");
        VmProgram firstProgram = firstSession.Decorate(new([firstOperation], new Dictionary<string,int>(), "common"));
        VmProgram secondProgram = secondSession.Decorate(new([secondOperation], new Dictionary<string,int>(), "common"));
        secondSession.Command(new("detach"));
        using var firstContext = Context(1);
        using var secondContext = new VmRunContext(new("second",1,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        Task firstRun = firstProgram.ExecuteAsync(firstContext, timeout.Token).AsTask();
        await WaitForAsync(() => firstSession.Snapshot().State == "paused");
        await secondProgram.ExecuteAsync(secondContext, timeout.Token);
        Assert.Equal(1, secondOperation.Count);
        Assert.Equal(0, firstOperation.Count);
        firstSession.Command(new("detach"));
        await firstRun;
    }

    [Fact]
    public async Task PausedTypedArrayIsInspectedWithExplicitTruncation()
    {
        using var session = CreateSession();
        VmProgram program = session.Decorate(new([new CounterOp("value")],new Dictionary<string,int>(),"common"));
        using var context = Context(1);
        context.Set("results", Enumerable.Range(0, 300).ToArray());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        Task execution = program.ExecuteAsync(context,timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        DebugInspectionResult result = await session.InspectAsync(new("results",session.Snapshot().Revision),timeout.Token);
        using var json = System.Text.Json.JsonDocument.Parse(result.Data);
        Assert.True(json.RootElement.GetProperty("truncated").GetBoolean());
        Assert.Equal(300,json.RootElement.GetProperty("totalCount").GetInt32());
        Assert.Equal(256,json.RootElement.GetProperty("items").GetArrayLength());
        session.Command(new("detach"));
        await execution;
    }

    [Fact]
    public void InstructionCatalogResolvesGuardsAndProgramLocalLabels()
    {
        using var session = CreateSession();
        session.Decorate(new VmProgramBuilder("prepare")
            .Step(new BranchIfInstruction("skip", "done", _ => true))
            .Label("check")
            .Step(new OpConditional(new CounterOp("work"), _ => false, "available"))
            .Label("done")
            .Build());
        session.Decorate(new VmProgramBuilder("common")
            .Label("done")
            .Step(new BranchIfInstruction("again", "done", _ => false))
            .Build());

        DebugInstruction[] rows = session.Instructions;
        Assert.Equal(new DebugJump("done", 2), Assert.Single(rows[0].Flow!.Jumps));
        Assert.Equal("check", Assert.Single(rows[1].Labels!));
        Assert.Equal("available", Assert.Single(rows[1].Flow!.Guards));
        Assert.Equal("done", Assert.Single(rows[1].EndLabels!));
        Assert.Equal(new DebugJump("done", 0), Assert.Single(rows[2].Flow!.Jumps));
    }

    #endregion

    #region Fixtures

    static CameraDebugSession CreateSession() => new("camera", new PausedRegisterInspector(InferenceBackend.Cpu), "camera_warmup");
    static VmRunContext Context(long runId) => new(new("camera",runId,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());

    static async Task WaitForAsync(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while(!condition()) await Task.Delay(5,timeout.Token);
    }

    sealed class CounterOp(string name) : IOp
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create(name,"counter");
        public int Count { get; private set; }
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
        {
            context.Set(name,++Count);
            return ValueTask.FromResult(OpResult.Continue);
        }
    }

    sealed class DeviceOp : IOp, IHasExecutionDevice, IDisposable
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create("device","device");
        public bool IsGpuExecution => true;
        public string ExecutionDeviceName => "CUDA";
        public bool Disposed { get; private set; }
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken) => ValueTask.FromResult(OpResult.Continue);
        public void Dispose() => Disposed = true;
    }
    #endregion
}
