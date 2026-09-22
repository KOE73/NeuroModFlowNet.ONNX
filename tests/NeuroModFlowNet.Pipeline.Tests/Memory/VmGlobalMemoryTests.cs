using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.Tests.Memory;

public sealed class VmGlobalMemoryTests
{
    [Fact]
    public void Snapshot_ReturnsCurrentGlobalValues()
    {
        var memory = new VmGlobalMemory();
        memory.Set("tracker.startZone", 123);

        IReadOnlyDictionary<string, object> snapshot = memory.Snapshot();

        Assert.Contains("tracker.startZone", memory.Keys);
        Assert.Equal(123, snapshot["tracker.startZone"]);
    }
}
