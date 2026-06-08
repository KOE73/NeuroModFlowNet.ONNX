namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Raised when an instruction reports a controlled failure instead of throwing directly.
/// </summary>
public sealed class OpFailureException : Exception
{
    public OpFailureException(string instructionName, string reason, Exception? innerException = null)
        : base($"Instruction '{instructionName}' failed: {reason}", innerException)
    {
        InstructionName = instructionName;
        Reason = reason;
    }

    public string InstructionName { get; }

    public string Reason { get; }
}

