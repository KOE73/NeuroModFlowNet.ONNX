using System.Text.Json;
using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Tracking;

public sealed class PrivateTrackingSequenceIntegrationTests
{
    const string SequencePathVariable = "NMFN_PRIVATE_TRACKING_SEQUENCE_JSON";
    const string ConfigPathVariable = "NMFN_PRIVATE_TRACKING_CONFIG_JSON";
    const string VideoPathVariable = "NMFN_PRIVATE_TRACKING_VIDEO_PATH";
    const string ModelPathVariable = "NMFN_PRIVATE_TRACKING_MODEL_PATH";

    static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    [Fact]
    public async Task PrivateSequence_RunsTrackerPipeline_WhenExternalFixtureIsConfigured()
    {
        PrivateTrackingRunConfigFixture?[] runConfigs = LoadRunConfigsOrDefault();
        foreach(PrivateTrackingRunConfigFixture? runConfig in runConfigs)
            await RunPrivateSequenceAsync(runConfig);
    }

    static async Task RunPrivateSequenceAsync(PrivateTrackingRunConfigFixture? runConfig)
    {
        string sequencePath = GetRequiredExternalPathOrSkip(runConfig?.SequenceJson);
        ValidateOptionalExternalPath(runConfig?.VideoPath, VideoPathVariable);
        ValidateOptionalExternalPath(runConfig?.ModelPath, ModelPathVariable);

        string json = await File.ReadAllTextAsync(sequencePath, TestContext.Current.CancellationToken);
        PrivateTrackingSequenceFixture fixture = JsonSerializer.Deserialize<PrivateTrackingSequenceFixture>(
            json,
            SerializerOptions) ?? throw new InvalidOperationException($"Private tracking fixture '{sequencePath}' is empty.");

        Assert.NotEmpty(fixture.Frames);

        IouTrackerOptions trackerOptions = fixture.TrackerOptions?.ToOptions() ?? new IouTrackerOptions();
        VmProgram program = new VmProgramBuilder()
            .Step(new Op_Track(
                "track.det.in",
                "track.out",
                "tracker.private",
                "tracker.private",
                trackerOptions,
                startZoneKey: fixture.StartZone is null ? null : "tracker.startZone",
                endZoneKey: fixture.EndZone is null ? null : "tracker.endZone"))
            .Build();

        var globalMemory = new VmGlobalMemory();
        if(fixture.StartZone is not null)
            globalMemory.Set("tracker.startZone", fixture.StartZone.ToTrackRect());

        if(fixture.EndZone is not null)
            globalMemory.Set("tracker.endZone", fixture.EndZone.ToTrackRect());

        await using var controller = new VmController(
            new VmControllerOptions(
                "private-tracking",
                MaxInFlight: Math.Max(1, fixture.MaxInFlight),
                OutputKeys: ["track.out"]),
            program,
            globalMemory);

        var outputs = new List<TrackedObject[]>();
        foreach(PrivateTrackingFrameFixture frame in fixture.Frames.OrderBy(frame => frame.FrameIndex))
        {
            var inputs = new VmRunInputs()
                .Add("track.det.in", frame.Detections.Select((detection, index) => detection.ToTrackDetection(index)).ToArray());

            Assert.True(controller.TryStartRun(inputs, out VmRunHandle handle));
            VmRunOutcome outcome = await handle.Completion.WaitAsync(TestContext.Current.CancellationToken);

            Assert.Equal(VmRunStatus.Completed, outcome.Status);
            Assert.NotNull(outcome.Output);
            Assert.True(outcome.Output.TryGet("track.out", out TrackedObject[] tracks));
            outputs.Add(tracks);
        }

        AssertPrivateExpectations(fixture.Expect, outputs);
    }

    static PrivateTrackingRunConfigFixture?[] LoadRunConfigsOrDefault()
    {
        string? configPaths = Environment.GetEnvironmentVariable(ConfigPathVariable);
        if(string.IsNullOrWhiteSpace(configPaths))
            return [null];

        return configPaths
            .Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(LoadRunConfig)
            .ToArray();
    }

    static PrivateTrackingRunConfigFixture LoadRunConfig(string configPath)
    {
        if(!File.Exists(configPath))
            throw new FileNotFoundException($"Configured private tracking run config does not exist: {configPath}", configPath);

        string json = File.ReadAllText(configPath);
        return JsonSerializer.Deserialize<PrivateTrackingRunConfigFixture>(json, SerializerOptions)
            ?? throw new InvalidOperationException($"Private tracking run config '{configPath}' is empty.");
    }

    static string GetRequiredExternalPathOrSkip(string? configuredPath)
    {
        string? path = configuredPath ?? Environment.GetEnvironmentVariable(SequencePathVariable);
        if(string.IsNullOrWhiteSpace(path))
            Assert.Skip($"Private tracking integration is disabled. Set {ConfigPathVariable} or {SequencePathVariable} to an external JSON fixture path.");

        if(!File.Exists(path))
            throw new FileNotFoundException($"Configured private tracking fixture does not exist: {path}", path);

        return path;
    }

    static void ValidateOptionalExternalPath(string? configuredPath, string environmentVariable)
    {
        string? path = configuredPath ?? Environment.GetEnvironmentVariable(environmentVariable);
        if(string.IsNullOrWhiteSpace(path))
            return;

        if(!File.Exists(path))
            throw new FileNotFoundException($"Configured private tracking asset does not exist: {path}", path);
    }

    static void AssertPrivateExpectations(PrivateTrackingExpectationsFixture? expectations, IReadOnlyList<TrackedObject[]> outputs)
    {
        int producedTrackFrames = outputs.Count(frameTracks => frameTracks.Length > 0);
        int confirmedTrackCount = outputs
            .SelectMany(frameTracks => frameTracks)
            .Where(track => track.IsConfirmed)
            .Select(track => track.ConfirmedTrackId)
            .Distinct()
            .Count();
        int maxTrackId = outputs.SelectMany(frameTracks => frameTracks).Select(track => track.TrackId).DefaultIfEmpty(0).Max();

        if(expectations is null)
        {
            Assert.True(producedTrackFrames > 0, "Private tracking sequence produced no tracks.");
            return;
        }

        if(expectations.MinProducedTrackFrames is int minProducedTrackFrames)
            Assert.True(
                producedTrackFrames >= minProducedTrackFrames,
                $"Produced track frames {producedTrackFrames} < expected {minProducedTrackFrames}.");

        if(expectations.MinConfirmedTracks is int minConfirmedTracks)
            Assert.True(
                confirmedTrackCount >= minConfirmedTracks,
                $"Confirmed tracks {confirmedTrackCount} < expected {minConfirmedTracks}.");

        if(expectations.ExpectedConfirmedTracks is int expectedConfirmedTracks)
            Assert.Equal(expectedConfirmedTracks, confirmedTrackCount);

        if(expectations.MaxTrackId is int expectedMaxTrackId)
            Assert.True(maxTrackId <= expectedMaxTrackId, $"Max TrackId {maxTrackId} > expected {expectedMaxTrackId}.");

        if(expectations.ExpectedMaxTrackId is int expectedMaxTrackIdExact)
            Assert.Equal(expectedMaxTrackIdExact, maxTrackId);
    }

    sealed record PrivateTrackingRunConfigFixture(
        string? SequenceJson,
        string? VideoPath,
        string? ModelPath);

    sealed record PrivateTrackingSequenceFixture(
        int MaxInFlight,
        PrivateIouTrackerOptionsFixture? TrackerOptions,
        PrivateTrackRectFixture? StartZone,
        PrivateTrackRectFixture? EndZone,
        PrivateTrackingExpectationsFixture? Expect,
        PrivateTrackingFrameFixture[] Frames);

    sealed record PrivateIouTrackerOptionsFixture(
        float? IouThreshold,
        int? MinAge,
        int? MaxMissedFrames,
        int? MaxTrailLength,
        float? StartZoneContainmentThreshold)
    {
        public IouTrackerOptions ToOptions() =>
            new(
                IouThreshold ?? 0.5f,
                MinAge ?? 15,
                MaxMissedFrames ?? 10,
                MaxTrailLength ?? 50,
                StartZoneContainmentThreshold ?? 1f);
    }

    sealed record PrivateTrackRectFixture(float X, float Y, float Width, float Height)
    {
        public TrackRect ToTrackRect() => new(X, Y, X + Width, Y + Height);
    }

    sealed record PrivateTrackingExpectationsFixture(
        int? MinProducedTrackFrames,
        int? MinConfirmedTracks,
        int? ExpectedConfirmedTracks,
        int? MaxTrackId,
        int? ExpectedMaxTrackId);

    sealed record PrivateTrackingFrameFixture(
        int FrameIndex,
        PrivateTrackDetectionFixture[] Detections);

    sealed record PrivateTrackDetectionFixture(
        float X,
        float Y,
        float W,
        float H,
        float Angle,
        int ClassId,
        float Score,
        int? SourceIndex)
    {
        public TrackDetection ToTrackDetection(int fallbackSourceIndex) =>
            new(X, Y, W, H, Angle, ClassId, Score, SourceIndex ?? fallbackSourceIndex);
    }
}
