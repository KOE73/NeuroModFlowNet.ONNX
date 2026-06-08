using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Mutable execution state for one <see cref="OnnxModel"/>.
/// </summary>
/// <remarks>
/// This class keeps the old runner mechanics: persistent input/output tensors, one reusable IoBinding, direct
/// SetInput/SetOutput values, and cleanup after a prediction. It is intentionally separate from <see cref="OnnxModel"/>
/// so the model/session metadata can live long without owning per-run OrtValue state.
/// </remarks>
[DebuggerDisplay("{DebuggerDisplay,nq}")]
[DebuggerTypeProxy(typeof(OnnxExecutionContextDebugView))]
public class OnnxExecutionContext : IDisposable, IOnnxModelOutputs
{
    readonly bool ownsModel;
    readonly Dictionary<string, OrtValue> runInputValues = new();
    readonly Dictionary<string, OrtValue> runOutputValues = new();
    readonly Dictionary<string, (long[] shape, OrtValue val)> inputPersistentValues = new();
    readonly Dictionary<string, (long[] shape, OrtValue val)> outputPersistentValues = new();
    bool disposed;

    public OnnxExecutionContext(OnnxModel model, bool ownsModel = false)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        this.ownsModel = ownsModel;

        RunOptions = new RunOptions();
        IoBinding = Model.Session.CreateIoBinding();
    }

    public OnnxModel Model { get; }

    public RunOptions RunOptions { get; }

    public OrtIoBinding IoBinding { get; }

    public void SetInput(string name, OrtValue value) => runInputValues[name] = value;

    public void SetOutput(string name, OrtValue value) => runOutputValues[name] = value;

    public void Cleanup()
    {
        foreach(OrtValue value in runInputValues.Values)
            value.Dispose();
        runInputValues.Clear();

        foreach(OrtValue value in runOutputValues.Values)
            value.Dispose();
        runOutputValues.Clear();
    }

    public void Run()
    {
        foreach(string name in Model.InputNames)
        {
            if(runInputValues.TryGetValue(name, out OrtValue? value))
                IoBinding.BindInput(name, value);
            else
                IoBinding.BindInput(name, GetInputPersistentValue(name));
        }

        foreach(string name in Model.OutputNames)
        {
            if(runOutputValues.TryGetValue(name, out OrtValue? value))
                IoBinding.BindOutput(name, value);
            else
                IoBinding.BindOutput(name, GetOutputPersistentValue(name));
        }

        Model.Session.RunWithBinding(RunOptions, IoBinding);
    }

    public OrtValue GetOutputValue(string? name = null) => GetOutputPersistentValue(name ?? Model.PrimaryOutputName);

    public ReadOnlySpan<T> GetTensorDataAsSpan<T>(string? name = null) where T : unmanaged =>
        GetOutputPersistentValue(name ?? Model.PrimaryOutputName).GetTensorDataAsSpan<T>();

    public long[] GetOutputShape(string? name = null) => GetRealOutputShape(name ?? Model.PrimaryOutputName);

    public long[] GetRealInputShape(string name) => inputPersistentValues[name].shape;

    public Span<T> GetInputBuffer<T>(string name) where T : unmanaged => GetInputPersistentValue(name).GetTensorMutableDataAsSpan<T>();

    public OrtValue InitInputPersistentValue(string name, long[] shape)
    {
        long[] modelShape = Model.ModelInputShapes[name];
        bool isCompatible = modelShape.Zip(shape, (modelDimension, actualDimension) => modelDimension <= 0 || modelDimension == actualDimension).All(static item => item);
        if(!isCompatible)
            throw new InvalidOperationException($"Shape {string.Join(",", shape)} is incompatible with model shape {string.Join(",", modelShape)}");

        OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, Model.GetInputElementType(name), shape);
        inputPersistentValues[name] = (shape, value);
        return value;
    }

    public bool IsInputPersistentValueInitialized(string name) => inputPersistentValues.ContainsKey(name);

    public long[] GetRealOutputShape(string name) => outputPersistentValues[name].shape;

    public Span<T> GetOutputBuffer<T>(string name) where T : unmanaged => GetOutputPersistentValue(name).GetTensorMutableDataAsSpan<T>();

    public OrtValue InitOutputPersistentValue(string name, long[] shape)
    {
        long[] modelShape = Model.ModelOutputShapes[name];
        bool isCompatible = modelShape.Zip(shape, (modelDimension, actualDimension) => modelDimension <= 0 || modelDimension == actualDimension).All(static item => item);
        if(!isCompatible)
            throw new InvalidOperationException($"Shape {string.Join(",", shape)} is incompatible with model shape {string.Join(",", modelShape)}");

        OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, Model.GetOutputElementType(name), shape);
        outputPersistentValues[name] = (shape, value);
        return value;
    }

    public bool IsOutputPersistentValueInitialized(string name) => outputPersistentValues.ContainsKey(name);

    public void Dispose()
    {
        if(disposed)
            return;

        disposed = true;

        Cleanup();

        foreach((long[] _, OrtValue value) in inputPersistentValues.Values)
            value.Dispose();
        inputPersistentValues.Clear();

        foreach((long[] _, OrtValue value) in outputPersistentValues.Values)
            value.Dispose();
        outputPersistentValues.Clear();

        IoBinding.Dispose();
        RunOptions.Dispose();

        if(ownsModel)
            Model.Dispose();
    }

    OrtValue GetInputPersistentValue(string name)
    {
        if(inputPersistentValues.TryGetValue(name, out (long[] shape, OrtValue value) item))
            return item.value;

        long[] shape = Model.ModelInputShapes[name];

        Debug.Assert(shape.Any(static dimension => dimension > 0), $"Cannot create persistent buffer for dynamic shape in '{name}'. Please provide OrtValue explicitly. Use method SetInput or Init for static shapes.");

        return InitInputPersistentValue(name, shape);
    }

    OrtValue GetOutputPersistentValue(string name)
    {
        if(outputPersistentValues.TryGetValue(name, out (long[] shape, OrtValue value) item))
            return item.value;

        long[] shape = Model.ModelOutputShapes[name];

        Debug.Assert(shape.Any(static dimension => dimension > 0), $"Cannot create persistent buffer for dynamic shape in '{name}'. Please provide OrtValue explicitly. Use method SetInput or Init for static shapes.");

        return InitOutputPersistentValue(name, shape);
    }

    string DebuggerDisplay =>
        $"{Model.InferenceBackend} {Model.ModelSource.DisplayName} | IN:{Model.InputNames.Count} OUT:{Model.OutputNames.Count}";
}
