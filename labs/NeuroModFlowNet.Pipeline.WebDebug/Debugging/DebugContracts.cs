namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed record DebugInstruction(string Program, int Index, string Name, string Operation, string[] Reads, string[] Writes);
internal sealed record DebugBreakpoint(string Program, int Index);
internal sealed record DebugLocation(string Program, int Index, string Name, string Phase, long RunId, long? FrameId);
internal sealed record DebugVariable(string Key, string Type, string Memory, string? Shape, string? ElementType, string? Value,
    bool CanInspectValue = false, bool CanPreviewImage = false);
internal sealed record DebugTiming(string Program, int Index, string Name, double Milliseconds);
internal sealed record DebugSnapshot(string State, long Revision, DebugLocation? Location, DebugVariable[] Variables,
    DebugBreakpoint[] Breakpoints, DebugTiming[] Timings, string? Error, DebugBreakpoint? RunTo = null);
internal sealed record DebugCommand(string Command, long? Revision = null, string? Program = null, int? Index = null);
internal sealed record DebugInspectionRequest(string Key, long Revision, string Format = "json", int MaxEdge = 1000);
internal sealed record DebugInspectionResult(string ContentType, byte[] Data);

internal sealed record DebugWatchSelection(string Key, string Format);
internal sealed record DebugWatchConfiguration(bool Enabled, double FramesPerSecond, int MaxEdge, DebugWatchSelection[] Registers);
internal sealed record DebugWatchTarget(string Key, string Format, string Program, int Index, string Phase);
internal sealed record DebugWatchAcknowledgement(long Generation, Dictionary<string, long> After);
internal sealed record DebugWatchFrame(long Sequence, DebugLocation Location, DateTimeOffset CapturedAt,
    string? ContentType, byte[]? Data, string? Error);
internal sealed record DebugWatchItem(DebugWatchTarget Target, long Captured, long Skipped, DebugWatchFrame? Frame);
internal sealed record DebugWatchSnapshot(long Generation, DebugWatchConfiguration Configuration, DebugWatchItem[] Items, DebugVariable[]? Variables = null);
