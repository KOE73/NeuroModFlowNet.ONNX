namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Legacy constructor facade over <see cref="OnnxExecutionContext"/>.
/// </summary>
/// <remarks>
/// New code should prefer creating an <see cref="OnnxModel"/> explicitly and then passing it to
/// <see cref="OnnxExecutionContext"/>. This type is kept as a thin bridge for existing factories, labs, and samples that
/// still create one object and expect it to own both the model session and the execution state.
/// </remarks>
public sealed class OnnxRuntimeContext : OnnxExecutionContext
{
    public static implicit operator OnnxRuntimeContext(string modelPath) => new(modelPath);

    public OnnxRuntimeContext(
        string modelPath,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null)
        : base(new OnnxModel(modelPath, inferenceBackend, configure), ownsModel: true)
    {
    }

    public OnnxRuntimeContext(
        byte[] modelBytes,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null,
        string? displayName = null)
        : base(new OnnxModel(modelBytes, inferenceBackend, configure, displayName), ownsModel: true)
    {
    }

    public OnnxRuntimeContext(
        ReadOnlyMemory<byte> modelBytes,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null,
        string? displayName = null)
        : base(new OnnxModel(modelBytes, inferenceBackend, configure, displayName), ownsModel: true)
    {
    }

    public OnnxRuntimeContext(
        OnnxRuntimeModelSource modelSource,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null)
        : base(new OnnxModel(modelSource, inferenceBackend, configure), ownsModel: true)
    {
    }

    public OnnxRuntimeContext(OnnxModel model)
        : base(model)
    {
    }
}
