namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Implemented by VM instructions that execute on a specific compute device (CPU or GPU).
/// Used by <see cref="DeviceCompatibilityValidator"/> to verify memory placement without reflection.
/// </summary>
public interface IHasExecutionDevice
{
    bool IsGpuExecution { get; }

    string ExecutionDeviceName { get; }
}
