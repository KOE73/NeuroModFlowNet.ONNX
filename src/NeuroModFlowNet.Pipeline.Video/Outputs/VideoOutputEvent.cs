namespace NeuroModFlowNet.Pipeline.Video;

public sealed record VideoOutputEvent(string StreamId, string OutputId, string State, string? Detail = null);
