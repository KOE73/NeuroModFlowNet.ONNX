using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Long-lived ONNX model container.
/// </summary>
/// <remarks>
/// The model owns the expensive ONNX Runtime session and immutable model metadata. It deliberately does not own
/// per-run <see cref="OrtValue"/> instances, input/output buffers, or binding state. Execution state belongs to
/// <see cref="OnnxExecutionContext"/>, which can be recreated without reloading model metadata contracts.
/// </remarks>
public sealed class OnnxModel : IDisposable, IModelMetadataProvider
{
    bool disposed;

    public OnnxModel(
        string modelPath,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null)
        : this(OnnxRuntimeModelSource.FromFile(modelPath), inferenceBackend, configure)
    {
    }

    public OnnxModel(
        byte[] modelBytes,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null,
        string? displayName = null)
        : this(OnnxRuntimeModelSource.FromBytes(modelBytes, displayName), inferenceBackend, configure)
    {
    }

    public OnnxModel(
        ReadOnlyMemory<byte> modelBytes,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null,
        string? displayName = null)
        : this(OnnxRuntimeModelSource.FromBytes(modelBytes, displayName), inferenceBackend, configure)
    {
    }

    public OnnxModel(
        OnnxRuntimeModelSource modelSource,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null)
    {
        ModelSource = modelSource;
        InferenceBackend = inferenceBackend;
        ModelPath = modelSource.Path ?? modelSource.DisplayName;

        using SessionOptions sessionOptions = CreateSessionOptions(inferenceBackend, configure);
        Session = modelSource.CreateSession(sessionOptions);

        InputNames = Session.InputNames;
        OutputNames = Session.OutputNames;

        ModelInputShapes = Session.InputMetadata
            .Where(item => item.Value.IsTensor)
            .ToDictionary(item => item.Key, item => (long[])[.. item.Value.Dimensions]);

        ModelOutputShapes = Session.OutputMetadata
            .Where(item => item.Value.IsTensor)
            .ToDictionary(item => item.Key, item => (long[])[.. item.Value.Dimensions]);
    }

    public InferenceBackend InferenceBackend { get; }

    public OnnxRuntimeModelSource ModelSource { get; }

    public string ModelPath { get; }

    public InferenceSession Session { get; }

    public IReadOnlyList<string> InputNames { get; }

    public IReadOnlyList<string> OutputNames { get; }

    public IReadOnlyDictionary<string, long[]> ModelInputShapes { get; }

    public IReadOnlyDictionary<string, long[]> ModelOutputShapes { get; }

    public string PrimaryInputName => InputNames[0];

    public string PrimaryOutputName => OutputNames[0];

    public TensorElementType GetInputElementType(string name) => Session.InputMetadata[name].ElementDataType;

    public TensorElementType GetOutputElementType(string name) => Session.OutputMetadata[name].ElementDataType;

    public string? GetCustomMetadata(string key)
        => Session.ModelMetadata.CustomMetadataMap.TryGetValue(key, out string? value) ? value : null;

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;
        Session.Dispose();
    }

    static SessionOptions CreateSessionOptions(InferenceBackend inferenceBackend, Action<ExecutionProviderConfig>? configure)
    {
        SessionOptions sessionOptions;
        switch(inferenceBackend)
        {
            case InferenceBackend.Rocm:
                sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider(0);
                DisableCpuEpFallback(sessionOptions);
                break;
            case InferenceBackend.Cuda:
                {
                    using var cudaOptions = new OrtCUDAProviderOptions();
                    CudaConfig cudaConfig = CudaConfig.FromDefaultOptions(cudaOptions);
                    configure?.Invoke(cudaConfig);
                    cudaOptions.UpdateOptions(cudaConfig);
                    sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaOptions);
                    ApplyExecutionProviderSessionOptions(sessionOptions, cudaConfig);
                    break;
                }
            case InferenceBackend.TensorRt:
                {
                    using var trtOptions = new OrtTensorRTProviderOptions();
                    TrtConfig trtConfig = TrtConfig.FromDefaultOptions(trtOptions);
                    if(configure != null) configure(trtConfig);
                    else
                    {
                        trtConfig.MaxWorkspaceSizeGb = 4;
                        trtConfig.EnableFp16 = true;
                        trtConfig.EnableBf16 = false;
                        trtConfig.EnableEngineCache = true;
                        trtConfig.EngineCachePath = TrtConfigDefaults.GetEngineCachePath();
                        trtConfig.BuilderOptimizationLevel = 2;
                    }
                    trtOptions.UpdateOptions(trtConfig);
                    sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtOptions);
                    ApplyExecutionProviderSessionOptions(sessionOptions, trtConfig);
                    break;
                }
            case InferenceBackend.DML:
                sessionOptions = new SessionOptions();
                sessionOptions.AppendExecutionProvider_DML(0);
                DisableCpuEpFallback(sessionOptions);
                break;
            default:
                sessionOptions = new SessionOptions();
                break;
        }
        return sessionOptions;
    }

    static void ApplyExecutionProviderSessionOptions(SessionOptions sessionOptions, ExecutionProviderConfig config)
    {
        if(config.DisableCpuEpFallback)
            DisableCpuEpFallback(sessionOptions);
    }

    static void DisableCpuEpFallback(SessionOptions sessionOptions)
    {
        sessionOptions.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
}
