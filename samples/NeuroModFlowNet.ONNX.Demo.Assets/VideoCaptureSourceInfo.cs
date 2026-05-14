namespace NeuroModFlowNet.ONNX.Demo.Assets;

public readonly record struct VideoCaptureSourceInfo(
    string ConfigKey,
    string ConfigValue,
    string ResolvedSource,
    string? LinkPath,
    string? LinkContent)
{
    public bool IsLinked => LinkPath is not null;
}
