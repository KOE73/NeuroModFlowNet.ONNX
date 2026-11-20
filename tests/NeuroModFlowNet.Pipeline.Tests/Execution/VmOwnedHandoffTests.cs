namespace NeuroModFlowNet.Pipeline.Tests;

public sealed class VmOwnedHandoffTests
{
    [Fact]
    public async Task PostMovesOwnershipBeforeContextIsDisposed()
    {
        var resource = new Resource();
        using var handoff = new VmOwnedHandoff<Resource>();
        await using(var run = CreateRun())
        {
            run.Set("value", resource, disposeWithContext: true);
            run.Set("handoff", handoff);
            var post = new Op_PostOwned<Resource>("value", "handoff");
            await post.ExecuteAsync(run, TestContext.Current.CancellationToken);
            Assert.False(run.Contains("value"));
        }

        Assert.Equal(0, resource.DisposeCount);
        handoff.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task WithoutHandoffTheContextStillDisposesTheRegister()
    {
        var resource = new Resource();
        await using(var run = CreateRun())
        {
            run.Set("value", resource, disposeWithContext: true);
            var post = new Op_PostOwned<Resource>("value", "handoff");
            await post.ExecuteAsync(run, TestContext.Current.CancellationToken);
            Assert.Same(resource, run.Get<Resource>("value"));
        }

        Assert.Equal(1, resource.DisposeCount);
    }

    static VmRunContext CreateRun() => new(new VmRunIdentity("source", 1, DateTimeOffset.UtcNow), new(), new());

    sealed class Resource : IDisposable
    {
        public int DisposeCount { get; private set; }
        public void Dispose() => DisposeCount++;
    }
}
