namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Resolved ONNX-related asset source.
/// </summary>
public readonly record struct OnnxAssetSource
{
    readonly byte[]? bytes;

    private OnnxAssetSource(OnnxAssetSourceKind kind, string displayName, string? path, byte[]? bytes)
    {
        Kind = kind;
        DisplayName = displayName;
        Path = path;
        this.bytes = bytes;
    }

    public OnnxAssetSourceKind Kind { get; }

    public string DisplayName { get; }

    public string? Path { get; }

    public bool IsPathBacked => Kind == OnnxAssetSourceKind.File;

    public static OnnxAssetSource FromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return new OnnxAssetSource(OnnxAssetSourceKind.File, System.IO.Path.GetFileName(path), path, null);
    }

    public static OnnxAssetSource FromBytes(byte[] bytes, string? displayName = null)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if(bytes.Length == 0)
            throw new ArgumentException("Asset byte array must not be empty.", nameof(bytes));

        return new OnnxAssetSource(
            OnnxAssetSourceKind.Bytes,
            string.IsNullOrWhiteSpace(displayName) ? "<memory>" : displayName,
            null,
            bytes);
    }

    public ReadOnlyMemory<byte> GetBytes()
    {
        if(Kind != OnnxAssetSourceKind.Bytes)
            throw new InvalidOperationException("Path-backed asset does not keep bytes in memory.");

        return bytes!;
    }

    public OnnxRuntimeModelSource ToModelSource()
    {
        return Kind switch
        {
            OnnxAssetSourceKind.File => OnnxRuntimeModelSource.FromFile(Path!),
            OnnxAssetSourceKind.Bytes => OnnxRuntimeModelSource.FromBytes(bytes!, DisplayName),
            _ => throw new InvalidOperationException($"Unsupported asset source kind: {Kind}")
        };
    }
}
