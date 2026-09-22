using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.Tests.Execution;

/// <summary>
/// EN: Verifies that a controller executes a chain of programs as one run with one shared register file.
/// RU: Проверяет, что контроллер выполняет цепочку программ как один запуск с общим регистровым файлом.
/// </summary>
public sealed class VmControllerProgramChainTests
{
    [Fact]
    public async Task Chain_SharesRunContextBetweenPrepareAndCommonPrograms()
    {
        VmProgram prepare = new VmProgramBuilder("prepare")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.write", "test.write", writes: [VarRequirement.Write<string>("image.model")]),
                (context, _) =>
                {
                    context.Set("image.model", "prepared-frame");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        VmProgram common = new VmProgramBuilder("common")
            .Step(new OpDelegate(
                OpDescriptor.Create("common.read", "test.read", reads: [VarRequirement.Read<string>("image.model")]),
                (context, _) =>
                {
                    context.Set("result", context.Get<string>("image.model") + ":processed");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", OutputKeys: ["result"]),
            [prepare, common]);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.Equal("prepared-frame:processed", outcome.Output!.Get<string>("result"));
        Assert.Equal(["prepare", "common"], outcome.Trace!.Instructions.Select(entry => entry.Program));
    }

    [Fact]
    public async Task Chain_StopInFirstProgram_SkipsRemainingPrograms()
    {
        bool commonExecuted = false;

        VmProgram prepare = new VmProgramBuilder("prepare")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.stop", "test.stop"),
                (_, _) => ValueTask.FromResult(OpResult.Stop)))
            .Build();

        VmProgram common = new VmProgramBuilder("common")
            .Step(new OpDelegate(
                OpDescriptor.Create("common.step", "test.step"),
                (_, _) =>
                {
                    commonExecuted = true;
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(new VmControllerOptions("source:0"), [prepare, common]);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.False(commonExecuted);
        Assert.Single(outcome.Trace!.Instructions);
    }

    [Fact]
    public async Task Chain_FailureInFirstProgram_FailsRunAndSkipsRemainingPrograms()
    {
        bool commonExecuted = false;

        VmProgram prepare = new VmProgramBuilder("prepare")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.fail", "test.fail"),
                (_, _) => ValueTask.FromResult(OpResult.Fail("prepare failed"))))
            .Build();

        VmProgram common = new VmProgramBuilder("common")
            .Step(new OpDelegate(
                OpDescriptor.Create("common.step", "test.step"),
                (_, _) =>
                {
                    commonExecuted = true;
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(new VmControllerOptions("source:0"), [prepare, common]);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Failed, outcome.Status);
        Assert.IsType<OpFailureException>(outcome.Exception);
        Assert.False(commonExecuted);
    }

    [Fact]
    public async Task Chain_LabelsAreLocalToEachProgram()
    {
        var executed = new List<string>();

        VmProgram prepare = new VmProgramBuilder("prepare")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.jump", "test.jump"),
                (_, _) => ValueTask.FromResult(OpResult.Jump("end"))))
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.skipped", "test.step"),
                (_, _) =>
                {
                    executed.Add("prepare.skipped");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Label("end")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.end", "test.step"),
                (_, _) =>
                {
                    executed.Add("prepare.end");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        VmProgram common = new VmProgramBuilder("common")
            .Label("end")
            .Step(new OpDelegate(
                OpDescriptor.Create("common.end", "test.step"),
                (_, _) =>
                {
                    executed.Add("common.end");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(new VmControllerOptions("source:0"), [prepare, common]);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.Equal(["prepare.end", "common.end"], executed);
    }

    [Fact]
    public async Task Chain_WarmupExecutesEveryProgram()
    {
        var executed = new List<string>();

        VmProgram prepare = new VmProgramBuilder("prepare")
            .Step(new OpDelegate(
                OpDescriptor.Create("prepare.step", "test.step"),
                (_, _) =>
                {
                    executed.Add("prepare");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        VmProgram common = new VmProgramBuilder("common")
            .Step(new OpDelegate(
                OpDescriptor.Create("common.step", "test.step"),
                (_, _) =>
                {
                    executed.Add("common");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", RequireWarmup: true),
            [prepare, common]);

        VmRunOutcome warmupOutcome = await controller.WarmupAsync(_ => { });

        Assert.Equal(VmRunStatus.Completed, warmupOutcome.Status);
        Assert.Equal(["prepare", "common"], executed);
        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle handle));
        Assert.Equal(VmRunStatus.Completed, (await handle.Completion).Status);
    }

    [Fact]
    public async Task Chain_DisposeReleasesInstructionsOfEveryProgram()
    {
        var prepareInstruction = new DisposableOp("prepare.step");
        var commonInstruction = new DisposableOp("common.step");

        VmProgram prepare = new VmProgramBuilder("prepare").Step(prepareInstruction).Build();
        VmProgram common = new VmProgramBuilder("common").Step(commonInstruction).Build();

        var controller = new VmController(new VmControllerOptions("source:0"), [prepare, common]);
        await controller.DisposeAsync();

        Assert.True(prepareInstruction.IsDisposed);
        Assert.True(commonInstruction.IsDisposed);
    }

    [Fact]
    public void Constructor_RejectsEmptyChain()
    {
        Assert.Throws<ArgumentException>(() =>
            new VmController(new VmControllerOptions("source:0"), Array.Empty<VmProgram>()));
    }

    sealed class DisposableOp(string name) : OpBase(OpDescriptor.Create(name, "test.disposable")), IDisposable
    {
        public bool IsDisposed { get; private set; }

        public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken) =>
            ValueTask.FromResult(OpResult.Continue);

        public void Dispose() => IsDisposed = true;
    }
}
