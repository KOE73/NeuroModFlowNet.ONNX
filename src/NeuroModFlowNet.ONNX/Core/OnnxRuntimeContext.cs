using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX;

public interface IOnnxModelOutputs
{
    OrtValue GetOutputValue(string? name = null);
    ReadOnlySpan<T> GetTensorDataAsSpan<T>(string? name = null) where T : unmanaged;
}

/// <summary>
/// EN: Base class for high-performance ONNX model inference. Encapsulates Session management, Hardware Acceleration (EP) configuration, and memory optimization via IoBinding. Allows viewing network input/output parameters in a convenient format and provides a simplified interface for single-I/O networks.
/// <br/><br/>
/// EN: The "Model" suffix in property names implies "as defined in the model", meaning it excludes current initialization state and potential runtime changes to input/output dimensions.
/// <br/><br/>
/// RU: Базовый класс для высокопроизводительного инференса ONNX-моделей. Инкапсулирует управление сессией, настройку аппаратного ускорения (EP) и оптимизацию памяти через IoBinding, позволяет просматривать входные и выходные параметры сети в удобном виде, имеет упрощенный интерфейс для сетей с одним входом и одним выходом.
/// <br/>
/// RU: Суффикс Model у свойств подразумевает "как в модели", то есть без учета текущей инициализации и возможных изменений размеров входов/выходов при работе.
/// <br/>
/// RU: .Where(kv => kv.Value.IsTensor) на случай если сети на входе например не тензоры имеют.
/// </summary>
/// <remarks>
/// <para>EN: Design Rationale:</para>
/// <list type="bullet">
/// <item><term>Performance</term><description> Uses IoBinding to minimize memory allocation overhead and allow direct memory access, reducing CPU-GPU transfer costs.</description></item>
/// <item><term>Flexibility</term><description> Supports multiple backends (CUDA, TensorRT, ROCm, DML) with optimized provider options and configurable settings during creation.</description></item>
/// <item><term>Introspection</term><description> Caches model metadata (shapes, names) to avoid redundant dictionary lookups during inference.</description></item>
/// </list>
/// <br/>
/// <para>RU: Обоснование архитектуры:</para>
/// <list type="bullet">
/// <item><term>Производительность</term><description> Использование IoBinding минимизирует накладные расходы на выделение памяти и позволяет работать с данными напрямую, исключая лишнее копирование данных между CPU и GPU.</description></item>
/// <item><term>Гибкость</term><description> Поддержка нескольких бэкендов (CUDA, TensorRT, ROCm, DML) с оптимизированными настройками провайдеров и возможностью управлять ими при создании.</description></item>
/// <item><term>Интроспекция</term><description> Кэширование метаданных (формы, имена) позволяет избежать лишних обращений к словарям во время инференса.</description></item>
/// </list>
/// </remarks>
[DebuggerDisplay("{DebuggerDisplay,nq}")]
[DebuggerTypeProxy(typeof(OnnxRuntimeContextDebugView))]
public sealed class OnnxRuntimeContext : IDisposable, IOnnxModelOutputs
{
    public static implicit operator OnnxRuntimeContext(string modelPath) => new(modelPath);

    /// <summary>
    /// EN: Initializes a new instance of the model with specified hardware acceleration.
    /// <br/>
    /// RU: Инициализирует новый экземпляр модели с выбранным типом аппаратного ускорения.
    /// </summary>
    /// <param name="modelPath">EN: Path to the .onnx file. RU: Путь к .onnx файлу.</param>
    /// <param name="inferenceBackend">EN: Target hardware (CUDA, TensorRT, etc.). RU: Целевое железо (CUDA, TensorRT и т.д.).</param>
    /// <param name="configure">EN: Optional custom configuration for the Execution Provider. RU: Опциональная кастомная настройка провайдера.</param>

    public OnnxRuntimeContext(
        string modelPath,
        InferenceBackend inferenceBackend = InferenceBackend.Cuda,
        Action<ExecutionProviderConfig>? configure = null)
    {
        InferenceBackend = inferenceBackend;
        ModelPath = modelPath;

        LoadModel(configure);
    }

    private void LoadModel(Action<ExecutionProviderConfig>? configure)
    {
        var options = MakeSessionOptions(InferenceBackend, configure);
        Session = new InferenceSession(ModelPath, options);

        InputNames = Session.InputNames;
        OutputNames = Session.OutputNames;

        ModelInputShapes = Session.InputMetadata
            .Where(kv => kv.Value.IsTensor)
            .ToDictionary(kv => kv.Key, kv => (long[])[.. kv.Value.Dimensions]);

        ModelOutputShapes = Session.OutputMetadata
            .Where(kv => kv.Value.IsTensor)
            .ToDictionary(kv => kv.Key, kv => (long[])[.. kv.Value.Dimensions]);

        RunOptions = new RunOptions();
        IoBinding = Session.CreateIoBinding();
    }

    private SessionOptions MakeSessionOptions(InferenceBackend inferenceBackend, Action<ExecutionProviderConfig>? configure)
    {
        SessionOptions sessionOptions;
        switch(inferenceBackend)
        {
            case InferenceBackend.Rocm:
                sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider(0);
                break;
            case InferenceBackend.Cuda:
                {
                    using var cudaOptions = new OrtCUDAProviderOptions();
                    CudaConfig cudaConfig = CudaConfig.FromDefaultOptions(cudaOptions);
                    configure?.Invoke(cudaConfig);
                    cudaOptions.UpdateOptions(cudaConfig);
                    sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaOptions);
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
                        trtConfig.EnableBf16 = true;
                        trtConfig.EnableEngineCache = true;
                        trtConfig.EngineCachePath = TrtConfigDefaults.GetEngineCachePath();
                        trtConfig.BuilderOptimizationLevel = 2;
                    }
                    trtOptions.UpdateOptions(trtConfig);
                    sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtOptions);
                    break;
                }
            case InferenceBackend.DML:
                sessionOptions = new SessionOptions();
                sessionOptions.AppendExecutionProvider_DML(0);
                break;
            default:
                sessionOptions = new SessionOptions();
                break;
        }
        return sessionOptions;
    }

    public void Dispose()
    {
        Session?.Dispose();
        RunOptions?.Dispose();
        IoBinding?.Dispose();
        foreach(var v in _InputPersistentValues.Values) v.val.Dispose();
        _InputPersistentValues.Clear();
        foreach(var v in _OutputPersistentValues.Values) v.val.Dispose();
        _OutputPersistentValues.Clear();
        Cleanup();
    }


    readonly Dictionary<string, OrtValue> RunInputValues = new();
    readonly Dictionary<string, OrtValue> RunOutputValues = new();

    /// <summary>
    /// Direct set input values.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="value"></param>
    public void SetInput(string name, OrtValue value) => RunInputValues[name] = value;
    public void SetOutput(string name, OrtValue value) => RunOutputValues[name] = value;

    public void Cleanup()
    {
        foreach(var v in RunInputValues.Values) v.Dispose();
        RunInputValues.Clear();
        foreach(var v in RunOutputValues.Values) v.Dispose();
        RunOutputValues.Clear();
    }

    public void Run()
    {
        // Bind all inputs
        foreach(var name in InputNames)
        {
            if(RunInputValues.TryGetValue(name, out var val))
                IoBinding.BindInput(name, val);
            else
                IoBinding.BindInput(name, GetInputPersistentValue(name));
        }

        // Bind all outputs
        foreach(var name in OutputNames)
        {
            if(RunOutputValues.TryGetValue(name, out var val))
                IoBinding.BindOutput(name, val);
            else
                IoBinding.BindOutput(name, GetOutputPersistentValue(name));
        }

        Session.RunWithBinding(RunOptions, IoBinding);
    }

    #region IOnnxModelOutputs

    public OrtValue GetOutputValue(string? name = null) => GetOutputPersistentValue(name ?? PrimaryOutputName);

    public ReadOnlySpan<T> GetTensorDataAsSpan<T>(string? name = null) where T : unmanaged => GetOutputPersistentValue(name ?? PrimaryOutputName).GetTensorDataAsSpan<T>();

    #endregion

    #region Input

    public long[] GetRealInputShape(string name) => _InputPersistentValues[name].shape;

    public Span<T> GetInputBuffer<T>(string name) where T : unmanaged => GetInputPersistentValue(name).GetTensorMutableDataAsSpan<T>();

    public OrtValue InitInputPersistentValue(string name, long[] shape)
    {
        var modelShape = ModelInputShapes[name];

        // Проверяем: если в модели размер > 0, он должен строго совпадать с входящим shape
        bool isCompatible = modelShape.Zip(shape, (m, s) => m <= 0 || m == s).All(x => x);
        if(!isCompatible) throw new InvalidOperationException($"Shape {string.Join(",", shape)} is incompatible with model shape {string.Join(",", modelShape)}");

        var meta = Session.InputMetadata[name];

        var val = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, meta.ElementDataType, shape);
        _InputPersistentValues[name] = (shape, val); 
        return val;
    }

    public bool IsInputPersistentValueInitialized(string name) => _InputPersistentValues.ContainsKey(name);

    readonly Dictionary<string, (long[] shape, OrtValue val)> _InputPersistentValues = new();

    OrtValue GetInputPersistentValue(string name)
    {
        if(_InputPersistentValues.TryGetValue(name, out var item)) return item.val;

        var shape = ModelInputShapes[name];

        Debug.Assert(shape.Any(d => d > 0), $"Cannot create persistent buffer for dynamic shape in '{name}'. Please provide OrtValue explicitly. Use method SetInput or Init for static shapes.");

        return InitInputPersistentValue(name, shape);
    }
    #endregion

    #region Output

    public long[] GetRealOutputShape(string name) => _OutputPersistentValues[name].shape;

    public Span<T> GetOutputBuffer<T>(string name) where T : unmanaged => GetOutputPersistentValue(name).GetTensorMutableDataAsSpan<T>();

    public OrtValue InitOutputPersistentValue(string name, long[] shape)
    {
        var modelShape = ModelOutputShapes[name];

        // Проверяем: если в модели размер > 0, он должен строго совпадать с входящим shape
        bool isCompatible = modelShape.Zip(shape, (m, s) => m <= 0 || m == s).All(x => x);
        if(!isCompatible) throw new InvalidOperationException($"Shape {string.Join(",", shape)} is incompatible with model shape {string.Join(",", modelShape)}");

        var meta = Session.OutputMetadata[name];
        var val = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, meta.ElementDataType, shape);
        _OutputPersistentValues[name] = (shape, val);
        return val;
    }

    public bool IsOutputPersistentValueInitialized(string name) => _OutputPersistentValues.ContainsKey(name);

    readonly Dictionary<string, (long[] shape, OrtValue val)> _OutputPersistentValues = new();

    OrtValue GetOutputPersistentValue(string name)
    {
        if(_OutputPersistentValues.TryGetValue(name, out var item)) return item.val;

        var shape = ModelOutputShapes[name];

        Debug.Assert(shape.Any(d => d > 0), $"Cannot create persistent buffer for dynamic shape in '{name}'. Please provide OrtValue explicitly. Use method SetInput or Init for static shapes.");

        return InitOutputPersistentValue(name, shape);
    }
    #endregion


    #region Metadata

    public InferenceBackend InferenceBackend { get; }
    public string ModelPath { get; }
    public InferenceSession Session { get; private set; } = default!;
    public RunOptions RunOptions { get; private set; } = default!;
    public OrtIoBinding IoBinding { get; private set; } = default!;
    public IReadOnlyList<string> InputNames { get; private set; } = default!;
    public IReadOnlyList<string> OutputNames { get; private set; } = default!;
    public Dictionary<string, long[]> ModelInputShapes { get; private set; } = default!;
    public Dictionary<string, long[]> ModelOutputShapes { get; private set; } = default!;

    public string PrimaryInputName => InputNames[0];
    public string PrimaryOutputName => OutputNames[0];

    public string? GetCustomMetadata(string key)
        => Session.ModelMetadata.CustomMetadataMap.TryGetValue(key, out var val) ? val : null;
    #endregion

    #region Debug Display

    string DebuggerDisplay =>
        $"{InferenceBackend} {Path.GetFileName(ModelPath)} | IN:{InputNames.Count} OUT:{OutputNames.Count}";

    #endregion
}
