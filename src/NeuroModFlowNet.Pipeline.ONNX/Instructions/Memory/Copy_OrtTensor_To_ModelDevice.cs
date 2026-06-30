using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Copies an ONNX Runtime tensor to the selected model execution provider memory using a dynamic Identity model.
/// </summary>
/// <remarks>
/// The instruction intentionally performs the transfer through ONNX Runtime itself. That keeps placement behavior under
/// the same execution-provider rules as real models and avoids baking CUDA-only copy code into the pipeline layer.
/// </remarks>
public sealed class Copy_OrtTensor_To_ModelDevice : OpBase, IDisposable
{
    readonly string inputKey;
    readonly string outputKey;
    readonly InferenceBackend executionBackend;
    OnnxExecutionContext? onnxContext;
    TensorElementType? initializedElementType;
    long[]? initializedInputShape;
    OrtMemoryInfo? cudaMemoryInfo;
    OrtAllocator? cudaAllocator;

    public Copy_OrtTensor_To_ModelDevice(string inputKey, string outputKey, InferenceBackend executionBackend = InferenceBackend.Cuda)
        : base(OpDescriptor.Create(
            "Copy_OrtTensor_To_ModelDevice",
            "copy.ortTensor.toModelDevice",
            reads: [VarRequirement.Read<OrtValue>(inputKey)],
            writes: [VarRequirement.Write<OrtValue>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.executionBackend = executionBackend;
    }

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(inputKey, out OrtValue inputOrtValue))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        TensorElementType elementType = inputOrtValue.GetTensorTypeAndShape().ElementDataType;
        long[] inputShape = inputOrtValue.GetTensorTypeAndShape().Shape;

        EnsureContext(elementType, inputShape);

        OrtValue outputOrtValue = CreateOutputOrtTensor(elementType, inputShape);

        // Binding a preallocated output is the important part: the Identity model computes nothing useful, but ORT must
        // materialize the result in the allocator selected by CreateOutputOrtTensor.
        RunWithFreshBinding(inputOrtValue, outputOrtValue);

        context.Set(outputKey, outputOrtValue, disposeWithContext: true);

        return ValueTask.FromResult(OpResult.Continue);
    }

    void EnsureContext(TensorElementType elementType, long[] inputShape)
    {
        if(onnxContext is not null &&
            initializedElementType == elementType &&
            initializedInputShape is not null &&
            initializedInputShape.SequenceEqual(inputShape))
        {
            return;
        }

        DisposeRuntimeState();

        RuntimeOnnxOperatorKernelKey kernelKey = new(
            "Copy_OrtTensor_To_ModelDevice",
            $"shape={FormatShape(inputShape)};type={elementType}",
            executionBackend,
            "runtime-operator;identity-upload");

        onnxContext = RuntimeOnnxOperatorKernelCache.CreateContext(
            kernelKey,
            () => IdentityBuilder.Build(ToTensorProtoDataType(elementType), inputShape),
            configure: null,
            "dynamic-identity.onnx");

        initializedElementType = elementType;
        initializedInputShape = [.. inputShape];

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

    OrtValue CreateOutputOrtTensor(TensorElementType elementType, long[] shape)
    {
        if(onnxContext!.Model.InferenceBackend == InferenceBackend.Cuda)
            return OrtValue.CreateAllocatedTensorValue(cudaAllocator!, elementType, shape);

        // CPU output is still allocated explicitly so the pipeline owns the destination OrtValue lifetime uniformly.
        return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, shape);
    }

    void RunWithFreshBinding(OrtValue inputOrtValue, OrtValue outputOrtValue)
    {
        ArgumentNullException.ThrowIfNull(onnxContext);

        // OrtIoBinding is native state owned by the cached session. It keeps references to bound OrtValue handles, so a
        // warmup binding must not survive into the production run after the warmup context has disposed its tensors.
        onnxContext.IoBinding.ClearBoundInputs();
        onnxContext.IoBinding.ClearBoundOutputs();

        try
        {
            onnxContext.IoBinding.BindInput(IdentityBuilder.InputName, inputOrtValue);
            onnxContext.IoBinding.BindOutput(IdentityBuilder.OutputName, outputOrtValue);
            onnxContext.Model.Session.RunWithBinding(onnxContext.RunOptions, onnxContext.IoBinding);
            onnxContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            onnxContext.IoBinding.ClearBoundInputs();
            onnxContext.IoBinding.ClearBoundOutputs();
        }
    }

    static Onnx.TensorProto.Types.DataType ToTensorProtoDataType(TensorElementType elementType) =>
        elementType switch
        {
            TensorElementType.UInt8 => Onnx.TensorProto.Types.DataType.Uint8,
            TensorElementType.Float => Onnx.TensorProto.Types.DataType.Float,
            TensorElementType.Float16 => Onnx.TensorProto.Types.DataType.Float16,
            _ => throw new NotSupportedException($"Unsupported tensor element type for dynamic Identity model: {elementType}")
        };

    static string FormatShape(long[] shape) => string.Join('x', shape);

    public void Dispose()
    {
        DisposeRuntimeState();
        initializedElementType = null;
        initializedInputShape = null;
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
}
