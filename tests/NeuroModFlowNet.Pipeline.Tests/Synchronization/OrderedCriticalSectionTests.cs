using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.Tests.Synchronization;

public sealed class OrderedCriticalSectionTests
{
    [Fact]
    public async Task SyncGateInstruction_ExecutesOneRunAtATimeInRunIdOrder()
    {
        var observedOrder = new List<long>();
        object syncRoot = new();
        int activeCount = 0;
        int maxActiveCount = 0;

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("delay-first", "test.delay"),
                async (context, cancellationToken) =>
                {
                    if(context.Identity.RunId == 0)
                        await Task.Delay(80, cancellationToken);

                    return OpResult.Continue;
                }))
            .Step(new OpDelegate(
                OpDescriptor.Create("critical", "test.critical", hasSideEffects: true, syncGate: "tracker"),
                async (context, cancellationToken) =>
                {
                    int currentActiveCount = Interlocked.Increment(ref activeCount);
                    maxActiveCount = Math.Max(maxActiveCount, currentActiveCount);

                    lock(syncRoot)
                        observedOrder.Add(context.Identity.RunId);

                    await Task.Delay(30, cancellationToken);
                    Interlocked.Decrement(ref activeCount);
                    return OpResult.Continue;
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
        Assert.Equal(1, maxActiveCount);
    }

    [Fact]
    public async Task SyncGateInstruction_ReleasesNextRunWhenOwnerFails()
    {
        var observedOrder = new List<long>();

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("critical", "test.critical", hasSideEffects: true, syncGate: "tracker"),
                (context, _) =>
                {
                    if(context.Identity.RunId == 0)
                        throw new InvalidOperationException("Synthetic failure inside critical section.");

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
    public async Task SyncGateInstruction_DoesNotWaitForeverWhenPreviousRunFailedBeforeGateWasCreated()
    {
        var observedOrder = new List<long>();

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("before-critical", "test.before-critical"),
                async (context, cancellationToken) =>
                {
                    if(context.Identity.RunId == 0)
                        throw new InvalidOperationException("Synthetic failure before critical section.");

                    await Task.Delay(80, cancellationToken);
                    return OpResult.Continue;
                }))
            .Step(new OpDelegate(
                OpDescriptor.Create("critical", "test.critical", hasSideEffects: true, syncGate: "tracker"),
                (context, _) =>
                {
                    observedOrder.Add(context.Identity.RunId);
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", MaxInFlight: 2),
            program);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle firstHandle));
        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle secondHandle));

        VmRunOutcome firstOutcome = await firstHandle.Completion;
        Task completedTask = await Task.WhenAny(
            secondHandle.Completion,
            Task.Delay(TimeSpan.FromSeconds(2), TestContext.Current.CancellationToken));

        Assert.Same(secondHandle.Completion, completedTask);

        VmRunOutcome secondOutcome = await secondHandle.Completion;

        Assert.Equal(VmRunStatus.Failed, firstOutcome.Status);
        Assert.Equal(VmRunStatus.Completed, secondOutcome.Status);
        Assert.Equal([1], observedOrder);
    }

    [Fact]
    public async Task InstructionWithoutSyncGate_KeepsExistingConcurrentBehavior()
    {
        int activeCount = 0;
        int maxActiveCount = 0;

        VmProgram program = new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create("plain", "test.plain", hasSideEffects: true),
                async (_, cancellationToken) =>
                {
                    int currentActiveCount = Interlocked.Increment(ref activeCount);
                    maxActiveCount = Math.Max(maxActiveCount, currentActiveCount);
                    await Task.Delay(50, cancellationToken);
                    Interlocked.Decrement(ref activeCount);
                    return OpResult.Continue;
                }))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", MaxInFlight: 2),
            program);

        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle firstHandle));
        Assert.True(controller.TryStartRun(_ => { }, out VmRunHandle secondHandle));

        VmRunOutcome[] outcomes = await Task.WhenAll(firstHandle.Completion, secondHandle.Completion);

        Assert.All(outcomes, outcome => Assert.Equal(VmRunStatus.Completed, outcome.Status));
        Assert.Equal(2, maxActiveCount);
    }
}
