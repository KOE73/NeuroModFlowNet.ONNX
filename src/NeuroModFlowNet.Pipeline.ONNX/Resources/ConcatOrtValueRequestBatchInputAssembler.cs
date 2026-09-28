using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using OnnxDataType = Onnx.TensorProto.Types.DataType;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class ConcatOrtValueRequestBatchInputAssembler : IOrtValueBatchInputAssembler
{
    OnnxExecutionContext? concatContext;
    InferenceBackend? initializedBackend;
    TensorElementType? initializedElementType;
    string? initializedShapeKey;
    OrtMemoryInfo? cudaMemoryInfo;
    OrtAllocator? cudaAllocator;

    public OrtValueBatchInput Assemble(
        IReadOnlyList<OrtValue> inputs,
        InferenceBackend executionBackend,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();

        if(inputs.Count == 0)
            throw new InvalidOperationException("Request-batch assembler received no inputs.");

        var firstInputInfo = inputs[0].GetTensorTypeAndShape();
        TensorElementType elementType = firstInputInfo.ElementDataType;
        long[][] inputShapes = new long[inputs.Count][];
        inputShapes[0] = firstInputInfo.Shape;
        int[] requestItemCounts = new int[inputs.Count];
        requestItemCounts[0] = ValidateInputShape(inputShapes[0], 0);

        for(int inputIndex = 1; inputIndex < inputs.Count; inputIndex++)
        {
            var inputInfo = inputs[inputIndex].GetTensorTypeAndShape();
            if(inputInfo.ElementDataType != elementType)
                throw new InvalidOperationException($"Input {inputIndex} element type differs from input 0: {inputInfo.ElementDataType} != {elementType}.");

            inputShapes[inputIndex] = inputInfo.Shape;
            requestItemCounts[inputIndex] = ValidateInputShape(inputShapes[inputIndex], inputIndex);
            ValidateTailShape(inputShapes[0], inputShapes[inputIndex], inputIndex);
        }

        int batchSize = requestItemCounts.Sum();
        if(inputs.Count == 1)
            return new OrtValueBatchInput(inputs[0], batchSize, OwnsValue: false, requestItemCounts);

        EnsureContext(inputShapes, elementType, executionBackend);

        long[] outputShape = [batchSize, .. inputShapes[0].Skip(1)];
        OrtValue batchValue = CreateOutputOrtTensor(elementType, outputShape);
        try
        {
            RunWithFreshBinding(inputs, batchValue);
            return new OrtValueBatchInput(batchValue, batchSize, OwnsValue: true, requestItemCounts);
        }
        catch
        {
            batchValue.Dispose();
            throw;
        }
    }

    static int ValidateInputShape(long[] inputShape, int inputIndex)
    {
        if(inputShape.Length == 0 || inputShape[0] <= 0)
            throw new InvalidOperationException($"Input {inputIndex} must have a positive batch dimension, actual shape: [{string.Join(", ", inputShape)}].");

        return checked((int)inputShape[0]);
    }

    static void ValidateTailShape(long[] expectedShape, long[] actualShape, int inputIndex)
    {
        if(actualShape.Length != expectedShape.Length)
            throw new InvalidOperationException($"Input {inputIndex} rank differs from input 0: {actualShape.Length} != {expectedShape.Length}.");

        for(int dimensionIndex = 1; dimensionIndex < expectedShape.Length; dimensionIndex++)
        {
            if(actualShape[dimensionIndex] != expectedShape[dimensionIndex])
                throw new InvalidOperationException($"Input {inputIndex} dimension {dimensionIndex} differs from input 0: {actualShape[dimensionIndex]} != {expectedShape[dimensionIndex]}.");
        }
    }

    void EnsureContext(long[][] inputShapes, TensorElementType elementType, InferenceBackend executionBackend)
    {
        string shapeKey = FormatShapes(inputShapes);
        if(concatContext is not null &&
            initializedBackend == executionBackend &&
            initializedElementType == elementType &&
            initializedShapeKey == shapeKey)
        {
            return;
        }

        DisposeContext();

        RuntimeOnnxOperatorKernelKey kernelKey = new(
            "RequestBatchConcat",
            $"shapes={shapeKey};type={elementType}",
            executionBackend,
            RuntimeOperatorProviderOptionsKey);

        concatContext = RuntimeOnnxOperatorKernelCache.CreateContext(
            kernelKey,
            () => BatchConcatBuilder.BuildVariableBatch(inputShapes, ToOnnxDataType(elementType)),
            ConfigureRuntimeOperatorExecutionProvider,
            "RequestBatchConcat");

        initializedBackend = executionBackend;
        initializedElementType = elementType;
        initializedShapeKey = shapeKey;

        if(executionBackend is InferenceBackend.Cuda or InferenceBackend.TensorRt)
        {
            cudaMemoryInfo = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA,
                OrtAllocatorType.DeviceAllocator,
                0,
                OrtMemType.Default);
            cudaAllocator = new OrtAllocator(concatContext.Model.Session, cudaMemoryInfo);
        }
    }

    static void ConfigureRuntimeOperatorExecutionProvider(ExecutionProviderConfig config)
    {
        if(config is TrtConfig trtConfig)
        {
            trtConfig.EnableBf16 = false;
        }
    }

    const string RuntimeOperatorProviderOptionsKey = "runtime-operator;trtEngineCache=keyed;trtBf16=false";

    static string FormatShapes(IReadOnlyList<long[]> shapes) =>
        string.Join('|', shapes.Select(static shape => string.Join('x', shape)));

    static OnnxDataType ToOnnxDataType(TensorElementType elementType) =>
        elementType switch
        {
            TensorElementType.UInt8 => OnnxDataType.Uint8,
            TensorElementType.Float => OnnxDataType.Float,
            TensorElementType.Float16 => OnnxDataType.Float16,
            _ => throw new NotSupportedException($"Request-batch concat does not support tensor element type: {elementType}.")
        };

    OrtValue CreateOutputOrtTensor(TensorElementType elementType, long[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(concatContext);

        return concatContext.Model.InferenceBackend switch
        {
            InferenceBackend.Cpu => OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, outputShape),
            InferenceBackend.Cuda or InferenceBackend.TensorRt => OrtValue.CreateAllocatedTensorValue(cudaAllocator!, elementType, outputShape),
            _ => throw new NotSupportedException($"Request-batch concat device output is not implemented for backend: {concatContext.Model.InferenceBackend}.")
        };
    }

    void RunWithFreshBinding(IReadOnlyList<OrtValue> inputs, OrtValue batchValue)
    {
        ArgumentNullException.ThrowIfNull(concatContext);

        concatContext.IoBinding.ClearBoundInputs();
        concatContext.IoBinding.ClearBoundOutputs();

        try
        {
            for(int slotIndex = 0; slotIndex < inputs.Count; slotIndex++)
                concatContext.IoBinding.BindInput(BatchConcatBuilder.InputName(slotIndex), inputs[slotIndex]);

            concatContext.IoBinding.BindOutput(BatchConcatBuilder.OutputName, batchValue);
            concatContext.Model.Session.RunWithBinding(concatContext.RunOptions, concatContext.IoBinding);
            concatContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            concatContext.IoBinding.ClearBoundInputs();
            concatContext.IoBinding.ClearBoundOutputs();
        }
    }

    void DisposeContext()
    {
        cudaAllocator?.Dispose();
        cudaAllocator = null;

        cudaMemoryInfo?.Dispose();
        cudaMemoryInfo = null;

        concatContext?.Dispose();
        concatContext = null;
    }

    public void Dispose()
    {
        DisposeContext();
        initializedBackend = null;
        initializedElementType = null;
        initializedShapeKey = null;
    }
}
