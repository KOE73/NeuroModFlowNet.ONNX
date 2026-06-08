namespace NeuroModFlowNet.Pipeline.VMDebug;

internal sealed record InstructionTimingSnapshot(
    string Name,
    double LastMilliseconds,
    double AverageMilliseconds);
