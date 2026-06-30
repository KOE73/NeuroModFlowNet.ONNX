using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Base class for VM instructions that execute a small generated ONNX graph over one tensor.
/// </summary>
/// <remarks>
/// These instructions are not domain model runners. They are pipeline operators: crop, resize, layout conversion,
/// normalization and other tensor-level steps that should obey the same execution-provider placement rules as the
/// following model. The base owns the repetitive ONNX Runtime mechanics so derived commands describe only the operation
/// contract: input validation, graph construction, output shape and output element type.
/// </remarks>
public abstract class Op_Onnx_TensorTransformBase : OpBase, IDisposable, IHasExecutionDevice
{
    readonly string inputKey;
    readonly string outputKey;
    readonly bool isFinal;
    readonly InferenceBackend executionBackend;
    long[]? initializedInputShape;
    TensorElementType? initializedInputElementType;
    OnnxExecutionContext? onnxContext;
    OrtMemoryInfo? cudaMemoryInfo;
    OrtAllocator? cudaAllocator;

    protected Op_Onnx_TensorTransformBase(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        bool isFinal,
        InferenceBackend? executionBackend = null)
        : base(descriptor)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.isFinal = isFinal;
        this.executionBackend = executionBackend ?? InferenceBackend.Cuda;
    }

    bool IHasExecutionDevice.IsGpuExecution =>
        onnxContext?.Model.InferenceBackend is { } backend && backend != InferenceBackend.Cpu;

    string IHasExecutionDevice.ExecutionDeviceName =>
        onnxContext?.Model.InferenceBackend.ToString() ?? "Uninitialized";

    protected abstract string GraphInputName { get; }

    protected abstract string GraphOutputName { get; }

    protected abstract string DisplayName { get; }

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(inputKey, out OrtValue inputOrtValue))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        var inputInfo = inputOrtValue.GetTensorTypeAndShape();
        long[] inputShape = inputInfo.Shape;
        TensorElementType inputElementType = inputInfo.ElementDataType;

        string? validationError = ValidateInput(inputShape, inputElementType);
        if(validationError is not null)
            return ValueTask.FromResult(OpResult.Fail(validationError));

        EnsureContext(inputShape, inputElementType);

        TensorElementType outputElementType = GetOutputElementType(inputElementType);
        long[] outputShape = CreateOutputShape(inputShape, inputElementType);
        OrtValue outputOrtValue = CreateOutputOrtTensor(outputElementType, outputShape);

        RunWithFreshBinding(inputOrtValue, outputOrtValue);

        context.Set(outputKey, outputOrtValue, disposeWithContext: true);
        WriteAdditionalOutputs(context, inputShape, outputShape);

        return ValueTask.FromResult(OpResult.Continue);
    }

    protected virtual string? ValidateInput(long[] inputShape, TensorElementType inputElementType) => null;

    protected abstract byte[] BuildModel(long[] inputShape, TensorElementType inputElementType);

    protected abstract long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType);

    protected virtual TensorElementType GetOutputElementType(TensorElementType inputElementType) => inputElementType;

    protected virtual string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) => string.Empty;

    protected virtual void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
    }

    protected static string? ValidateNhwcImageShape(long[] inputShape, int expectedChannels)
    {
        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";

        if(inputShape[0] != 1)
            return $"Input tensor batch must be 1, actual batch: {inputShape[0]}.";

        if(inputShape[3] != expectedChannels)
            return $"Input tensor channel count must be {expectedChannels}, actual channels: {inputShape[3]}.";

        return null;
    }

    void EnsureContext(long[] inputShape, TensorElementType inputElementType)
    {
        if(onnxContext is not null &&
            initializedInputElementType == inputElementType &&
            initializedInputShape is not null &&
            initializedInputShape.SequenceEqual(inputShape))
        {
            return;
        }

        DisposeRuntimeState();

        RuntimeOnnxOperatorKernelKey kernelKey = new(
            DisplayName,
            $"input={FormatShape(inputShape)};inputType={inputElementType};output={FormatShape(CreateOutputShape(inputShape, inputElementType))};outputType={GetOutputElementType(inputElementType)};op={GetCacheSemanticKey(inputShape, inputElementType)}",
            executionBackend,
            RuntimeOperatorProviderOptionsKey);

        onnxContext = RuntimeOnnxOperatorKernelCache.CreateContext(
            kernelKey,
            () => BuildModel(inputShape, inputElementType),
            ConfigureRuntimeOperatorExecutionProvider,
            DisplayName);

        initializedInputShape = [.. inputShape];
        initializedInputElementType = inputElementType;

        if(onnxContext.Model.InferenceBackend == InferenceBackend.Cuda)
        {
            cudaMemoryInfo = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA,
                OrtAllocatorType.DeviceAllocator,
                0,
                OrtMemType.Default);
            cudaAllocator = new OrtAllocator(onnxContext.Model.Session, cudaMemoryInfo);
        }
    }

    static void ConfigureRuntimeOperatorExecutionProvider(ExecutionProviderConfig config)
    {
        if(config is TrtConfig trtConfig)
            trtConfig.EnableEngineCache = false;
    }

    const string RuntimeOperatorProviderOptionsKey = "runtime-operator;trtEngineCache=false";

    static string FormatShape(long[] shape) => string.Join('x', shape);

    OrtValue CreateOutputOrtTensor(TensorElementType outputElementType, long[] outputShape)
    {
        if(!isFinal && onnxContext!.Model.InferenceBackend == InferenceBackend.Cuda)
            return OrtValue.CreateAllocatedTensorValue(cudaAllocator!, outputElementType, outputShape);

        // Final operator outputs and explicit CPU outputs are host-readable. Intermediate CUDA outputs stay in provider
        // memory so the following ONNX operator/model can consume them without an implicit upload.
        return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, outputElementType, outputShape);
    }

    void RunWithFreshBinding(OrtValue inputOrtValue, OrtValue outputOrtValue)
    {
        ArgumentNullException.ThrowIfNull(onnxContext);

        // IoBinding is native mutable state owned by the cached session. Per-run tensors are short-lived VM resources,
        // so stale bindings must never survive across warmup/production frames or across failed executions.
        onnxContext.IoBinding.ClearBoundInputs();
        onnxContext.IoBinding.ClearBoundOutputs();

        try
        {
            onnxContext.IoBinding.BindInput(GraphInputName, inputOrtValue);
            onnxContext.IoBinding.BindOutput(GraphOutputName, outputOrtValue);
            onnxContext.Model.Session.RunWithBinding(onnxContext.RunOptions, onnxContext.IoBinding);
            onnxContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            onnxContext.IoBinding.ClearBoundInputs();
            onnxContext.IoBinding.ClearBoundOutputs();
        }
    }

    void DisposeRuntimeState()
    {
        cudaAllocator?.Dispose();
        cudaAllocator = null;

        cudaMemoryInfo?.Dispose();
        cudaMemoryInfo = null;

        onnxContext?.Dispose();
        onnxContext = null;
    }

    public void Dispose()
    {
        DisposeRuntimeState();
        initializedInputShape = null;
        initializedInputElementType = null;
    }
}
