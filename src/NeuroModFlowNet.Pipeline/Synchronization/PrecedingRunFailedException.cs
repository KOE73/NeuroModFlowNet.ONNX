namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Thrown at a synchronization gate when the run that was supposed to precede this one
/// closed without reaching the gate (failed or canceled before this point).
/// </summary>
/// <remarks>
/// This is distinct from <see cref="OperationCanceledException"/>: the current run was not
/// canceled by the host — it was skipped because an earlier run did not complete this gate.
/// The VM controller marks the current run as <see cref="VmRunStatus.Failed"/> so downstream
/// observers can distinguish this case from a user-initiated cancellation.
/// </remarks>
public sealed class PrecedingRunFailedException : Exception
{
    public PrecedingRunFailedException(long skippedRunId)
        : base($"Run {skippedRunId} could not proceed: the preceding run was closed before reaching this synchronization point.")
    {
        SkippedRunId = skippedRunId;
    }

    public long SkippedRunId { get; }
}
