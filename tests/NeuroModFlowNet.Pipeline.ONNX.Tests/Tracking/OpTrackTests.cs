using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Tracking;

public sealed class OpTrackTests
{
    [Fact]
    public async Task Program_ConvertsYoloObbAndTracksWithSourceDetectionIndex()
    {
        VmProgram program = new VmProgramBuilder()
            .Step(new Op_DetectionsFromYoloObb("yolo.obb", "track.det.in"))
            .Step(new Op_Track(
                "track.det.in",
                "track.out",
                "tracker.obb",
                "tracker",
                new IouTrackerOptions(IouThreshold: 0.2f, MinAge: 1, MaxMissedFrames: 2, MaxTrailLength: 10),
                startZoneKey: "tracker.startZone"))
            .Build();
        var globalMemory = new VmGlobalMemory();
        globalMemory.Set("tracker.startZone", new TrackRect(0, 0, 100, 100));

        await using var controller = new VmController(
            new VmControllerOptions("source:0", OutputKeys: ["track.out"]),
            program,
            globalMemory);

        var inputs = new VmRunInputs()
            .Add("yolo.obb", new YoloDetectionBatchResult<YoloObb>(
            [
                new YoloObb { X = 50, Y = 50, W = 20, H = 20, Angle = 0, Class = 7, Score = 0.9f },
                new YoloObb { X = 140, Y = 140, W = 20, H = 20, Angle = 0, Class = 7, Score = 0.8f }
            ]));

        Assert.True(controller.TryStartRun(inputs, out VmRunHandle handle));
        VmRunOutcome outcome = await handle.Completion;

        Assert.Equal(VmRunStatus.Completed, outcome.Status);
        Assert.NotNull(outcome.Output);
        Assert.True(outcome.Output.TryGet("track.out", out TrackedObject[] tracks));
        TrackedObject track = Assert.Single(tracks);
        Assert.Equal(0, track.SourceDetectionIndex);
        Assert.Equal(7, track.ClassId);
        Assert.True(track.IsConfirmed);
    }

    [Fact]
    public async Task OpTrack_WithSyncGate_IsDeterministicUnderConcurrentRuns()
    {
        VmProgram program = new VmProgramBuilder()
            .Step(new Op_DetectionsFromYoloObb("yolo.obb", "track.det.in"))
            .Step(new Op_Track(
                "track.det.in",
                "track.out",
                "tracker.obb",
                "tracker",
                new IouTrackerOptions(IouThreshold: 0.2f, MinAge: 2, MaxMissedFrames: 2, MaxTrailLength: 10)))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("source:0", MaxInFlight: 2, OutputKeys: ["track.out"]),
            program);

        Assert.True(controller.TryStartRun(CreateInputs(50), out VmRunHandle firstHandle));
        Assert.True(controller.TryStartRun(CreateInputs(52), out VmRunHandle secondHandle));

        VmRunOutcome[] outcomes = await Task.WhenAll(firstHandle.Completion, secondHandle.Completion);

        Assert.All(outcomes, outcome => Assert.Equal(VmRunStatus.Completed, outcome.Status));
        Assert.True(outcomes[0].Output!.TryGet("track.out", out TrackedObject[] firstTracks));
        Assert.True(outcomes[1].Output!.TryGet("track.out", out TrackedObject[] secondTracks));
        Assert.False(Assert.Single(firstTracks).IsConfirmed);
        TrackedObject secondTrack = Assert.Single(secondTracks);
        Assert.Equal(1, secondTrack.TrackId);
        Assert.True(secondTrack.IsConfirmed);
        Assert.Equal(1, secondTrack.ConfirmedTrackId);
    }

    static VmRunInputs CreateInputs(float x) =>
        new VmRunInputs()
            .Add("yolo.obb", new YoloDetectionBatchResult<YoloObb>(
            [
                new YoloObb { X = x, Y = 50, W = 20, H = 20, Angle = 0, Class = 0, Score = 0.9f }
            ]));
}
