using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class RunToInstructionTests
{
    #region Run-to scenarios
    [Fact]
    public async Task RunToStopsBeforeTargetAcrossProgramsWithoutChangingBreakpoints()
    {
        using var session = Session();
        var prepareOperation = new CounterOp("prepared");
        var commonOperation = new CounterOp("common");
        VmProgram prepare = Program(session,"prepare",prepareOperation);
        VmProgram common = Program(session,"common",commonOperation);
        session.SetBreakpoints([new("prepare",0)]);
        using var context = Context(1);
        using var timeout = Timeout();
        Task execution = Chain();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        session.Command(new("run-to",session.Snapshot().Revision,"common",0));
        await WaitForAsync(() => session.Snapshot().Location?.Program == "common");
        Assert.Equal(1,prepareOperation.Count);
        Assert.Equal(0,commonOperation.Count);
        Assert.Equal("before",session.Snapshot().Location!.Phase);
        Assert.Null(session.Snapshot().RunTo);
        Assert.Equal(new DebugBreakpoint("prepare",0),Assert.Single(session.Snapshot().Breakpoints));
        session.Command(new("detach"));
        await execution;

        async Task Chain()
        {
            await prepare.ExecuteAsync(context,timeout.Token);
            await common.ExecuteAsync(context,timeout.Token);
        }
    }

    [Fact]
    public async Task ExistingBreakpointInterruptsAndClearsTemporaryTarget()
    {
        using var session = Session();
        var target = new CounterOp("target");
        VmProgram program = Program(session,"common",new CounterOp("first"),new CounterOp("middle"),target);
        session.SetBreakpoints([new("common",1)]);
        using var context = Context(1);
        using var timeout = Timeout();
        Task execution = program.ExecuteAsync(context,timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        session.Command(new("run-to",session.Snapshot().Revision,"common",2));
        await WaitForAsync(() => session.Snapshot().Location?.Index == 1);
        Assert.Equal(0,target.Count);
        Assert.Null(session.Snapshot().RunTo);
        session.SetBreakpoints([]);
        session.Command(new("continue"));
        await execution.WaitAsync(timeout.Token);
        Assert.Equal(1,target.Count);
    }

    [Fact]
    public async Task InvalidTargetAndAlreadyReachedTargetDoNotResume()
    {
        using var session = Session();
        var operation = new CounterOp("value");
        VmProgram program = Program(session,"common",operation);
        using var context = Context(1);
        using var timeout = Timeout();
        Task execution = program.ExecuteAsync(context,timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        Assert.Throws<ArgumentException>(() => session.Command(new("run-to",session.Snapshot().Revision,"missing",0)));
        Assert.Throws<ArgumentException>(() => session.Command(new("run-to",session.Snapshot().Revision,"common",10)));
        long revision = session.Snapshot().Revision;
        session.Command(new("run-to",revision,"common",0));
        Assert.Equal(revision,session.Snapshot().Revision);
        Assert.Equal("paused",session.Snapshot().State);
        Assert.Equal(0,operation.Count);
        session.Command(new("detach"));
        await execution;
    }

    [Fact]
    public async Task PassedTargetStopsBeforeItsNextFrameOccurrence()
    {
        using var session = Session();
        var operation = new CounterOp("value");
        VmProgram program = Program(session,"common",operation);
        using var context = Context(1);
        using var nextContext = Context(2);
        using var timeout = Timeout();
        Task firstRun = program.ExecuteAsync(context,timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().State == "paused");
        session.Command(new("step"));
        await WaitForAsync(() => session.Snapshot().Location?.Phase == "after");
        session.Command(new("run-to",session.Snapshot().Revision,"common",0));
        await firstRun;
        Task nextRun = program.ExecuteAsync(nextContext,timeout.Token).AsTask();
        await WaitForAsync(() => session.Snapshot().Location?.RunId == 2);
        Assert.Equal("before",session.Snapshot().Location!.Phase);
        Assert.Equal(1,operation.Count);
        Assert.False(nextContext.Contains("value"));
        session.Command(new("detach"));
        await nextRun;
    }

    #endregion

    #region Fixtures

    static CameraDebugSession Session() => new("camera",new PausedRegisterInspector(InferenceBackend.Cpu,Nv12ColorMatrix.Bt601Full));
    static VmRunContext Context(long run) => new(new("camera",run,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());
    static VmProgram Program(CameraDebugSession session,string name,params IOp[] instructions) => session.Decorate(new(instructions,new Dictionary<string,int>(),name));
    static CancellationTokenSource Timeout() => new(TimeSpan.FromSeconds(10));

    static async Task WaitForAsync(Func<bool> condition)
    {
        using var timeout = Timeout();
        while(!condition()) await Task.Delay(5,timeout.Token);
    }

    sealed class CounterOp(string key) : IOp
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create(key,"counter");
        public int Count { get; private set; }
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context,CancellationToken cancellationToken)
        {
            context.Set(key,++Count);
            return ValueTask.FromResult(OpResult.Continue);
        }
    }
    #endregion
}
