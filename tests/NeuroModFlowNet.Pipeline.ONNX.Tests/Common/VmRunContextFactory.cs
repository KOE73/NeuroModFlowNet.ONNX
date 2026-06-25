using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class VmRunContextFactory
{
    public static VmRunContext Create() =>
        new(
            new VmRunIdentity("pipeline-onnx-test", 0, DateTimeOffset.UtcNow),
            new VmGlobalMemory(),
            new VmSyncGateRegistry());
}
