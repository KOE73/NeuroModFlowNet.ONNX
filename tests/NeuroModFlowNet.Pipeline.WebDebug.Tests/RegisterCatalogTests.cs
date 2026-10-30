using System.Text;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class RegisterCatalogTests
{
    #region Catalog lifetime and boundaries

    [Fact]
    public void RetainsRemovedNamesAcrossFramesWithoutKeepingValues()
    {
        var inspector = Inspector();
        var catalog = new RegisterCatalog((context, key) => inspector.Describe(context, key, includeValue: false));
        catalog.Register(new("common", 0, "write", "test", [], ["work"]));
        catalog.Register(new("common", 1, "remove", "test", ["work"], []));
        catalog.Renew();
        using(var first = Context(1))
        {
            first.Set("work", "sensitive-payload");
            catalog.Visit(first, catalog.Boundary("common", 0, "after"));
            first.Remove("work");
            catalog.Visit(first, catalog.Boundary("common", 1, "after"));
        }
        DebugVariable removed = Assert.Single(catalog.Snapshot().Variables);
        Assert.Equal("removed", removed.Presence);
        Assert.Equal(1, removed.RemovedAt!.Index);
        Assert.Null(removed.Value);
        Assert.All(removed.Points!, point => Assert.Null(point.Variable.Value));
        using var second = Context(2);
        second.Set("other", 4);
        catalog.Visit(second, catalog.Boundary("common", 0, "before"));
        DebugVariable retained = catalog.Snapshot().Variables.Single(variable => variable.Key == "work");
        Assert.Equal("previous-run", retained.Presence);
        Assert.Equal(0, catalog.Resolve(new("work", "json", "common", 0, "after")).Index);
        Assert.DoesNotContain("sensitive-payload", System.Text.Json.JsonSerializer.Serialize(catalog.Snapshot()));
    }

    [Fact]
    public void AdditionOrderFollowsDiscoveryAndSurvivesOverwriteRemovalAndNextFrame()
    {
        var catalog = new RegisterCatalog((context, key) => new(key, "int", "Managed", null, null, null));
        catalog.Register(new("common", 0, "op", "test", [], []));
        catalog.Renew();
        using(var first = Context(1))
        {
            first.Set("z-first", 1);
            catalog.Visit(first, catalog.Boundary("common", 0, "before"));
            first.Set("a-second", 2); first.Set("z-first", 3);
            catalog.Visit(first, catalog.Boundary("common", 0, "after"));
            first.Remove("z-first");
            catalog.Visit(first, catalog.Boundary("common", 0, "before"));
        }
        using var second = Context(2);
        second.Set("z-first", 4); second.Set("late-branch", 5);
        catalog.Visit(second, catalog.Boundary("common", 0, "after"));
        catalog.RememberPause([new("a-second", "int", "Managed", null, null, null)], new("common", 0, "op", "after", 2, null));
        var variables = catalog.Snapshot().Variables.ToDictionary(variable => variable.Key);
        Assert.Equal(1, variables["z-first"].AdditionOrder);
        Assert.Equal(2, variables["a-second"].AdditionOrder);
        Assert.Equal(3, variables["late-branch"].AdditionOrder);
    }

    [Fact]
    public void CreatorRequiresObservedAbsenceAndIsNotReplacedByAnOverwrite()
    {
        var catalog = new RegisterCatalog((context, key) => new(key, "int", "Managed", null, null, null));
        catalog.Register(new("common", 0, "overwrite-input", "test", [], ["input"]));
        catalog.Register(new("common", 1, "create", "test", [], ["work"]));
        catalog.Register(new("common", 2, "overwrite", "test", [], ["work"]));
        catalog.Renew();
        using(var late = Context(1))
        {
            late.Set("work", 1);
            catalog.Visit(late, catalog.Boundary("common", 2, "after"));
            Assert.Null(Assert.Single(catalog.Snapshot().Variables).CreatedAt);
        }
        using var next = Context(2);
        next.Set("input", 3);
        catalog.Visit(next, catalog.Boundary("common", 0, "before"));
        next.Set("input", 4);
        catalog.Visit(next, catalog.Boundary("common", 0, "after"));
        catalog.Visit(next, catalog.Boundary("common", 1, "before"));
        next.Set("work", 5);
        catalog.Visit(next, catalog.Boundary("common", 1, "after"));
        catalog.Visit(next, catalog.Boundary("common", 2, "before"));
        next.Set("work", 6);
        catalog.Visit(next, catalog.Boundary("common", 2, "after"));
        var variables = catalog.Snapshot().Variables.ToDictionary(variable => variable.Key);
        Assert.Null(variables["input"].CreatedAt);
        Assert.Equal(1, variables["work"].CreatedAt!.Index);
        Assert.Equal("create", variables["work"].CreatedAt!.Name);
        Assert.Equal(1, variables["work"].AdditionOrder);
    }

    [Fact]
    public void DisabledAndExpiredCatalogDoesNotEnumerateOrDescribeOrAllocate()
    {
        var clock = new ManualClock();
        int descriptions = 0;
        var catalog = new RegisterCatalog((context, key) => { descriptions++; return new(key, "int", "Managed", null, null, null); }, clock);
        catalog.Register(new("common", 0, "write", "test", [], ["value"]));
        RegisterCatalogBoundary boundary = catalog.Boundary("common", 0, "after");
        using var context = Context(1);
        context.Set("value", 1);
        catalog.Visit(context, boundary);
        Assert.Equal(0, descriptions);
        catalog.Renew();
        catalog.Visit(context, boundary);
        Assert.Equal(1, descriptions);
        clock.Timestamp += 6000;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for(int index = 0; index < 10000; index++) catalog.Visit(context, boundary);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(1, descriptions);
        Assert.False(catalog.Snapshot().Collecting);
    }

    [Fact]
    public void MissingAcrossLeaseGapOrNewRunIsNotReportedAsInstructionRemoval()
    {
        var clock = new ManualClock();
        var catalog = new RegisterCatalog((context, key) => new(key, "int", "Managed", null, null, null), clock);
        catalog.Register(new("common", 0, "op", "test", [], []));
        using var context = Context(1);
        context.Set("value", 1);
        catalog.Renew(); catalog.Visit(context, catalog.Boundary("common", 0, "before"));
        clock.Timestamp += 6000;
        context.Remove("value");
        catalog.Renew(); catalog.Visit(context, catalog.Boundary("common", 0, "after"));
        Assert.Null(Assert.Single(catalog.Snapshot().Variables).RemovedAt);
    }

    [Fact]
    public void CapacityAndMetadataFailuresRemainVisibleWithoutFailingTheVm()
    {
        var catalog = new RegisterCatalog((context, key) => throw new ObjectDisposedException(key), capacity: 1);
        catalog.Register(new("common", 0, "op", "test", [], []));
        using var context = Context(1);
        context.Set("first", 1); context.Set("second", 2);
        catalog.Renew(); catalog.Visit(context, catalog.Boundary("common", 0, "before"));
        var snapshot = catalog.Snapshot();
        Assert.True(snapshot.Truncated);
        Assert.NotNull(Assert.Single(snapshot.Variables).MetadataError);
    }

    #endregion

    #region Reused names and exact dynamic observation

    [Fact]
    public async Task TwoViewersSampleDifferentWritesOfOneNameBeforeItIsRemoved()
    {
        using var session = new CameraDebugSession("camera", Inspector());
        VmProgram program = session.Decorate(new([new WriteOp(1), new WriteOp(2), new RemoveOp()], new Dictionary<string,int>(), "common"));
        session.RegisterCatalogSnapshot(renew: true);
        session.Command(new("detach"));
        using(var first = Context(1)) await program.ExecuteAsync(first, TestContext.Current.CancellationToken);
        DebugVariable variable = session.RegisterCatalogSnapshot().Variables.Single(variable => variable.Key == "work");
        Assert.Equal(0, variable.CreatedAt!.Index);
        Assert.Contains(variable.Points!, point => point.Index == 0 && point.Phase == "after" && point.IsWrite);
        Assert.Contains(variable.Points!, point => point.Index == 1 && point.Phase == "after" && point.IsWrite);
        DebugWatchSnapshot firstViewer = session.ConfigureObservation(new(true, 15, 640, [new("work", "json", "common", 0, "after")]), "first");
        DebugWatchSnapshot secondViewer = session.ConfigureObservation(new(true, 15, 640, [new("work", "json", "common", 1, "after")]), "second");
        DebugWatchSnapshot absentViewer = session.ConfigureObservation(new(true, 15, 640, [new("work", "json", "common", 2, "after")]), "absent");
        session.Observation.Poll(new(firstViewer.Generation, []), "first");
        session.Observation.Poll(new(secondViewer.Generation, []), "second");
        session.Observation.Poll(new(absentViewer.Generation, []), "absent");
        using var second = Context(2);
        await program.ExecuteAsync(second, TestContext.Current.CancellationToken);
        Assert.False(second.Contains("work"));
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while(session.Observation.Snapshot(viewerId: "first").Items[0].Captured == 0 || session.Observation.Snapshot(viewerId: "second").Items[0].Captured == 0)
            await Task.Delay(5, timeout.Token);
        var earlier = session.Observation.Poll(new(firstViewer.Generation, []), "first").Items[0].Frame!;
        var later = session.Observation.Poll(new(secondViewer.Generation, []), "second").Items[0].Frame!;
        Assert.Equal("1", Encoding.UTF8.GetString(earlier.Data!));
        Assert.Equal("2", Encoding.UTF8.GetString(later.Data!));
        Assert.Equal(0, earlier.Location.Index); Assert.Equal(1, later.Location.Index);
        Assert.Equal(2, earlier.Location.RunId); Assert.Equal(2, later.Location.RunId);
        Assert.Equal(0, session.RegisterCatalogSnapshot().Variables.Single(variable => variable.Key == "work").CreatedAt!.Index);
        var absent = session.Observation.Poll(new(absentViewer.Generation, []), "absent").Items[0].Frame!;
        Assert.Null(absent.Data);
        Assert.Contains("not present", absent.Error);
        Assert.Equal(2, absent.Location.Index);
        Assert.Throws<ArgumentException>(() => session.ConfigureObservation(new(true, 3, 640, [new("work", "json", "common", 99, "after")])));
    }

    [Fact]
    public void RemovingAliasDoesNotDisposeTheContextOwnedResource()
    {
        var resource = new OwnedResource();
        using(var context = Context(1))
        {
            context.Set("first", resource, disposeWithContext: true);
            context.Set("second", resource);
            Assert.True(context.Remove("first"));
            Assert.Same(resource, context.Get<OwnedResource>("second"));
            Assert.Equal(0, resource.Disposals);
        }
        Assert.Equal(1, resource.Disposals);
    }

    #endregion

    #region Fixtures

    static PausedRegisterInspector Inspector() => new(InferenceBackend.Cpu, Nv12ColorMatrix.Bt601Full);
    static VmRunContext Context(long run) => new(new("camera", run, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());
    sealed class ManualClock : TimeProvider
    {
        public long Timestamp { get; set; }
        public override long TimestampFrequency => 1000;
        public override long GetTimestamp() => Timestamp;
    }
    sealed class WriteOp(int value) : IOp
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create("write", "test", writes: [VarRequirement.Write<int>("work")]);
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
        {
            context.Set("work", value); return ValueTask.FromResult(OpResult.Continue);
        }
    }
    sealed class RemoveOp : IOp
    {
        public OpDescriptor Descriptor { get; } = OpDescriptor.Create("remove", "test", reads: [VarRequirement.Read<int>("work")]);
        public ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
        {
            context.Remove("work"); return ValueTask.FromResult(OpResult.Continue);
        }
    }
    sealed class OwnedResource : IDisposable
    {
        public int Disposals { get; private set; }
        public void Dispose() => Disposals++;
    }

    #endregion
}
