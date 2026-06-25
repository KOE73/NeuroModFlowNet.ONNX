namespace NeuroModFlowNet.Pipeline.Tests.Contexts;

public sealed class VmRunContextCoordinateTests
{
    [Fact]
    public void VmRunContext_DoesNotExposeGlobalCoordinateGraph()
    {
        Assert.Null(typeof(VmRunContext).GetProperty("Coordinates"));
    }

    [Fact]
    public void VmRunContext_CanStoreCoordinateBackTransformAsNamedRegister()
    {
        using var context = new VmRunContext(
            new VmRunIdentity("pipeline-test", 0, DateTimeOffset.UtcNow),
            new VmGlobalMemory(),
            new VmSyncGateRegistry());
        var transform = new CropCoordinateBackTransform(2, 3);

        context.Set("transform.crop.toSource", transform);

        Assert.True(context.TryGet("transform.crop.toSource", out ICoordinateBackTransform stored));
        Assert.Same(transform, stored);
    }
}
