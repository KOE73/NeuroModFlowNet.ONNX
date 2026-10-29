namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class DebugDeviceOp(IOp inner, CameraDebugSession session, string program, int index)
    : DebugOp(inner, session, program, index), IHasExecutionDevice
{
    public bool IsGpuExecution => ((IHasExecutionDevice)Inner).IsGpuExecution;
    public string ExecutionDeviceName => ((IHasExecutionDevice)Inner).ExecutionDeviceName;
}
