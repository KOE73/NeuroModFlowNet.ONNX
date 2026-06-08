namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Optional execution gate used by debuggers to stop before the next VM instruction.
/// </summary>
/// <remarks>
/// Production execution normally leaves this unset, so the VM loop pays only one nullable check per instruction.
/// Implementations must be non-blocking when no interactive debugging session is active.
/// </remarks>
public interface IOpDebugGate
{
    ValueTask WaitBeforeInstructionAsync(OpDebugPoint point, CancellationToken cancellationToken);
}
