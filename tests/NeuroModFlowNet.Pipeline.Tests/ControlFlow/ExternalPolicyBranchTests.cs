using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.Tests.ControlFlow;

public sealed class ExternalPolicyBranchTests
{
    [Fact]
    public async Task ExternalPolicy_CanSkipOcrBranch_WhenNoConfirmedTrackIsSelected()
    {
        int ocrExecutionCount = 0;
        VmProgram program = CreatePolicyDrivenOcrProgram(_ => false, () => ocrExecutionCount++);

        await using var controller = new VmController(
            new VmControllerOptions(
                "policy-test",
                MaxInFlight: 1,
                OutputKeys: ["ocr.shouldRun", "ocr.result"]),
            program);

        var inputs = new VmRunInputs()
            .Add("track.out", Array.Empty<TrackedObject>());

        Assert.True(controller.TryStartRun(inputs, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.NotNull(outcome.Output);
        Assert.True(outcome.Output.TryGet("ocr.shouldRun", out bool shouldRun));
        Assert.False(shouldRun);
        Assert.False(outcome.Output.TryGet("ocr.result", out string _));
        Assert.Equal(0, ocrExecutionCount);
    }

    [Fact]
    public async Task ExternalPolicy_CanSelectTrackAndRunOcrBranch()
    {
        int ocrExecutionCount = 0;
        TrackedObject selectedTrack = CreateConfirmedTrack(trackId: 7);
        VmProgram program = CreatePolicyDrivenOcrProgram(
            track => track.TrackId == selectedTrack.TrackId,
            () => ocrExecutionCount++);

        await using var controller = new VmController(
            new VmControllerOptions(
                "policy-test",
                MaxInFlight: 1,
                OutputKeys: ["ocr.shouldRun", "ocr.trackRect", "ocr.result"]),
            program);

        var inputs = new VmRunInputs()
            .Add("track.out", new[] { CreateConfirmedTrack(trackId: 3), selectedTrack });

        Assert.True(controller.TryStartRun(inputs, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.NotNull(outcome.Output);
        Assert.True(outcome.Output.TryGet("ocr.shouldRun", out bool shouldRun));
        Assert.True(shouldRun);
        Assert.True(outcome.Output.TryGet("ocr.trackRect", out TrackRect trackRect));
        Assert.Equal(selectedTrack.AxisAlignedBounds, trackRect);
        Assert.True(outcome.Output.TryGet("ocr.result", out string result));
        Assert.Equal("recognized-track-7", result);
        Assert.Equal(1, ocrExecutionCount);
    }

    static VmProgram CreatePolicyDrivenOcrProgram(
        Func<TrackedObject, bool> shouldRunOcr,
        Action onOcrExecuted)
    {
        return new VmProgramBuilder()
            .Step(new OpDelegate(
                OpDescriptor.Create(
                    "external-ocr-policy",
                    "policy.external.ocr",
                    reads: [VarRequirement.Read<TrackedObject[]>("track.out")],
                    writes:
                    [
                        VarRequirement.Write<bool>("ocr.shouldRun"),
                        VarRequirement.Write<TrackRect>("ocr.trackRect"),
                        VarRequirement.Write<int>("ocr.trackId")
                    ],
                    hasSideEffects: true),
                (context, _) =>
                {
                    TrackedObject? selectedTrack = context.Get<TrackedObject[]>("track.out")
                        .Where(static track => track.IsConfirmed && track.MissedFrames == 0)
                        .FirstOrDefault(shouldRunOcr);

                    if(selectedTrack is null)
                    {
                        context.Set("ocr.shouldRun", false);
                        return ValueTask.FromResult(OpResult.Continue);
                    }

                    context.Set("ocr.shouldRun", true);
                    context.Set("ocr.trackId", selectedTrack.TrackId);
                    context.Set("ocr.trackRect", selectedTrack.AxisAlignedBounds);
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Step(new BranchIfInstruction(
                "skip-ocr-when-policy-says-no",
                targetLabel: "after-ocr",
                condition: context => !context.Get<bool>("ocr.shouldRun"),
                reads: [VarRequirement.Read<bool>("ocr.shouldRun")]))
            .Step(new OpDelegate(
                OpDescriptor.Create(
                    "ocr-branch-placeholder",
                    "test.ocr.branch",
                    reads:
                    [
                        VarRequirement.Read<int>("ocr.trackId"),
                        VarRequirement.Read<TrackRect>("ocr.trackRect")
                    ],
                    writes: [VarRequirement.Write<string>("ocr.result")],
                    hasSideEffects: true),
                (context, _) =>
                {
                    onOcrExecuted();
                    int trackId = context.Get<int>("ocr.trackId");
                    context.Set("ocr.result", $"recognized-track-{trackId}");
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Label("after-ocr")
            .Build();
    }

    static TrackedObject CreateConfirmedTrack(int trackId) =>
        new(
            TrackId: trackId,
            ConfirmedTrackId: trackId,
            IsConfirmed: true,
            Age: 5,
            MissedFrames: 0,
            X: 100 + trackId,
            Y: 200 + trackId,
            W: 40,
            H: 60,
            Angle: 0,
            ClassId: 0,
            Score: 0.9f,
            SourceDetectionIndex: trackId,
            Path: []);
}
