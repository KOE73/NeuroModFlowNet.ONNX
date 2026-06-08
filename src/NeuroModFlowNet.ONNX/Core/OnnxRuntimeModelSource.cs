using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Describes where an ONNX Runtime session should load its model from.
/// </summary>
public readonly record struct OnnxRuntimeModelSource
{
    readonly byte[]? modelBytes;

    private OnnxRuntimeModelSource(
        OnnxRuntimeModelSourceKind kind,
        string displayName,
        string? path,
        byte[]? modelBytes)
    {
        Kind = kind;
        DisplayName = displayName;
        Path = path;
        this.modelBytes = modelBytes;
    }

    public OnnxRuntimeModelSourceKind Kind { get; }

    public string DisplayName { get; }

    public string? Path { get; }

    public bool IsPathBacked => Kind == OnnxRuntimeModelSourceKind.File;

    public static OnnxRuntimeModelSource FromFile(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        return new OnnxRuntimeModelSource(
            OnnxRuntimeModelSourceKind.File,
            System.IO.Path.GetFileName(modelPath),
            modelPath,
            null);
    }

    public static OnnxRuntimeModelSource FromBytes(byte[] modelBytes, string? displayName = null)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);

        if(modelBytes.Length == 0)
            throw new ArgumentException("Model byte array must not be empty.", nameof(modelBytes));

        return new OnnxRuntimeModelSource(
            OnnxRuntimeModelSourceKind.Bytes,
            string.IsNullOrWhiteSpace(displayName) ? "<memory>" : displayName,
            null,
            modelBytes);
    }

    public static OnnxRuntimeModelSource FromBytes(ReadOnlyMemory<byte> modelBytes, string? displayName = null)
    {
        if(modelBytes.IsEmpty)
            throw new ArgumentException("Model memory must not be empty.", nameof(modelBytes));

        return FromBytes(modelBytes.ToArray(), displayName);
    }

    internal InferenceSession CreateSession(SessionOptions sessionOptions)
    {
        return Kind switch
        {
            OnnxRuntimeModelSourceKind.File => new InferenceSession(Path!, sessionOptions),
            OnnxRuntimeModelSourceKind.Bytes => new InferenceSession(modelBytes!, sessionOptions),
            _ => throw new InvalidOperationException($"Unsupported ONNX model source kind: {Kind}")
        };
    }
}

