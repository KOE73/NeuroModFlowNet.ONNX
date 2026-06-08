namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Control-flow result returned by one VM instruction.
/// </summary>
public readonly record struct OpResult(
    OpResultKind Kind,
    string? Label = null,
    string? Reason = null,
    Exception? Exception = null)
{
    public static OpResult Continue { get; } = new(OpResultKind.Continue);

    public static OpResult Stop { get; } = new(OpResultKind.Stop);

    public static OpResult Jump(string label)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(label);
        return new OpResult(OpResultKind.Jump, Label: label);
    }

    public static OpResult Fail(string reason, Exception? exception = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reason);
        return new OpResult(OpResultKind.Fail, Reason: reason, Exception: exception);
    }
}

