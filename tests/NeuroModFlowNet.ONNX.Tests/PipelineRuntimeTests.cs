using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.ONNX.Tests;

public sealed class PipelineRuntimeTests
{
    [Fact]
    public async Task OrderedSyncInstruction_ReleasesRunsByAcceptedRunId()
    {
        var observedOrder = new List<long>();
        object observedOrderSyncRoot = new();

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("delay-first-run", "test.delay"),
                async (context, cancellationToken) =>
                {
                    if(context.Identity.RunId == 0)
                        await Task.Delay(80, cancellationToken);

                    return OpResult.Continue;
                }))
            .Step(new OrderedSyncInstruction("tracker"))
            .Step(new OpDelegate(
                OpDescriptor.Create("record-order", "test.record", hasSideEffects: true),
                (context, _) =>
                {
                    lock(observedOrderSyncRoot)
                        observedOrder.Add(context.Identity.RunId);

                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", MaxInFlight: 2),
            program);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle firstHandle));
        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle secondHandle));

        VmRunOutcome[] outcomes = await Task.WhenAll(firstHandle.Completion, secondHandle.Completion);

        Assert.All(outcomes, outcome => Assert.Equal(VmRunStatus.Completed, outcome.Status));
        Assert.Equal([0, 1], observedOrder);
    }

    [Fact]
    public async Task OrderedSyncInstruction_DoesNotWaitForeverForFailedPreviousRun()
    {
        var observedOrder = new List<long>();
        object observedOrderSyncRoot = new();

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("fail-first-run", "test.fail"),
                (context, _) =>
                {
                    if(context.Identity.RunId == 0)
                        throw new InvalidOperationException("Synthetic run failure before sync.");

                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Step(new OrderedSyncInstruction("tracker"))
            .Step(new OpDelegate(
                OpDescriptor.Create("record-order", "test.record", hasSideEffects: true),
                (context, _) =>
                {
                    lock(observedOrderSyncRoot)
                        observedOrder.Add(context.Identity.RunId);

                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", MaxInFlight: 2),
            program);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle firstHandle));
        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle secondHandle));

        VmRunOutcome[] outcomes = await Task.WhenAll(firstHandle.Completion, secondHandle.Completion);

        Assert.Equal(VmRunStatus.Failed, outcomes[0].Status);
        Assert.Equal(VmRunStatus.Completed, outcomes[1].Status);
        Assert.Equal([1], observedOrder);
    }

    [Fact]
    public async Task VmRunContext_DisposesOwnedResources()
    {
        var disposable = new DisposableProbe();
        var identity = new VmRunIdentity("source:0", 0, DateTimeOffset.UtcNow);
        await using var context = new VmRunContext(identity, new VmGlobalMemory(), new VmSyncGateRegistry());

        context.Set("resource", disposable, disposeWithContext: true);
        await context.DisposeAsync();

        Assert.True(disposable.Disposed);
    }

    [Fact]
    public async Task VmController_AcceptsNamedInputsAndCapturesRequestedOutputs()
    {
        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create(
                    "copy-input",
                    "test.copy",
                    [VarRequirement.Read<int>("input.value")],
                    [VarRequirement.Write<int>("output.value")]),
                (context, _) =>
                {
                    int value = context.Get<int>("input.value");
                    context.Set("output.value", value + 1);
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", OutputKeys: ["output.value"]),
            program);

        var inputs = new VmRunInputs()
            .Add("input.value", 41);

        Assert.True(controller.TryStartRun(inputs, out VmRunHandle handle));

        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.NotNull(outcome.Output);
        Assert.True(outcome.Output.TryGet("output.value", out int value));
        Assert.Equal(42, value);
        Assert.NotNull(outcome.Trace);
        Assert.Single(outcome.Trace.Instructions);
        Assert.Equal("test.copy", outcome.Trace.Instructions[0].Operation);
    }

    sealed class DisposableProbe : IDisposable
    {
        public bool Disposed { get; private set; }

        public void Dispose() => Disposed = true;
    }
}
