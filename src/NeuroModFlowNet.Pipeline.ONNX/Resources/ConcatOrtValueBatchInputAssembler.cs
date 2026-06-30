using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using OnnxDataType = Onnx.TensorProto.Types.DataType;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class ConcatOrtValueBatchInputAssembler : IOrtValueBatchInputAssembler
{
    readonly int? fixedBatchSize;
    OnnxExecutionContext? concatContext;
    InferenceBackend? initializedBackend;
    TensorElementType? initializedElementType;
    long[]? initializedSingleInputShape;
    int initializedBatchSize;
    int initializedBoundInputCount;
    OrtMemoryInfo? cudaMemoryInfo;
    OrtAllocator? cudaAllocator;

    public ConcatOrtValueBatchInputAssembler(int? fixedBatchSize = null)
    {
        if(fixedBatchSize is <= 1)
            throw new ArgumentOutOfRangeException(nameof(fixedBatchSize), "Fixed batch size must be greater than 1.");

        this.fixedBatchSize = fixedBatchSize;
    }

    public OrtValueBatchInput Assemble(
        IReadOnlyList<OrtValue> inputs,
        InferenceBackend executionBackend,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();

        if(inputs.Count == 0)
            throw new InvalidOperationException("Batch assembler received no inputs.");

        int effectiveBatchSize = fixedBatchSize ?? inputs.Count;
        if(effectiveBatchSize < inputs.Count)
            throw new InvalidOperationException($"Fixed batch size {effectiveBatchSize} is smaller than request count {inputs.Count}.");

        if(effectiveBatchSize == 1)
            return new OrtValueBatchInput(inputs[0], BatchSize: 1, OwnsValue: false);

        var firstInputInfo = inputs[0].GetTensorTypeAndShape();
        long[] singleInputShape = firstInputInfo.Shape;
        TensorElementType elementType = firstInputInfo.ElementDataType;

        ValidateInputs(inputs, singleInputShape, elementType);
        EnsureContext(effectiveBatchSize, inputs.Count, singleInputShape, elementType, executionBackend);

        long[] outputShape = [effectiveBatchSize, .. singleInputShape.Skip(1)];
        OrtValue batchValue = CreateOutputOrtTensor(elementType, outputShape);

        RunWithFreshBinding(inputs, batchValue);
        return new OrtValueBatchInput(batchValue, BatchSize: effectiveBatchSize, OwnsValue: true);
    }

    static void ValidateInputs(IReadOnlyList<OrtValue> inputs, long[] expectedShape, TensorElementType expectedElementType)
    {
        if(expectedShape.Length == 0 || expectedShape[0] != 1)
            throw new InvalidOperationException($"Each batch item must have batch dimension 1, actual shape: [{string.Join(", ", expectedShape)}].");

        for(int index = 1; index < inputs.Count; index++)
        {
            var inputInfo = inputs[index].GetTensorTypeAndShape();
            if(inputInfo.ElementDataType != expectedElementType)
                throw new InvalidOperationException($"Batch item {index} element type differs from item 0: {inputInfo.ElementDataType} != {expectedElementType}.");

            if(!inputInfo.Shape.SequenceEqual(expectedShape))
                throw new InvalidOperationException($"Batch item {index} shape differs from item 0: [{string.Join(", ", inputInfo.Shape)}] != [{string.Join(", ", expectedShape)}].");
        }
    }

    void EnsureContext(
        int batchSize,
        int boundInputCount,
        long[] singleInputShape,
        TensorElementType elementType,
        InferenceBackend executionBackend)
    {
        if(concatContext is not null &&
            initializedBackend == executionBackend &&
            initializedBatchSize == batchSize &&
            initializedBoundInputCount == boundInputCount &&
            initializedElementType == elementType &&
            initializedSingleInputShape is not null &&
            initializedSingleInputShape.SequenceEqual(singleInputShape))
        {
            return;
        }

        DisposeContext();

        byte[] modelBytes = BatchConcatBuilder.Build(batchSize, boundInputCount, singleInputShape, ToOnnxDataType(elementType));
        concatContext = new OnnxExecutionContext(new OnnxModel(modelBytes, executionBackend, ConfigureRuntimeOperatorExecutionProvider, "BatchConcat"), ownsModel: true);
        initializedBackend = executionBackend;
        initializedBatchSize = batchSize;
        initializedBoundInputCount = boundInputCount;
        initializedElementType = elementType;
        initializedSingleInputShape = [.. singleInputShape];

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
            trtConfig.EnableEngineCache = false;
    }

    static OnnxDataType ToOnnxDataType(TensorElementType elementType) =>
        elementType switch
        {
            TensorElementType.UInt8 => OnnxDataType.Uint8,
            TensorElementType.Float => OnnxDataType.Float,
            TensorElementType.Float16 => OnnxDataType.Float16,
            _ => throw new NotSupportedException($"Batch concat does not support tensor element type: {elementType}.")
        };

    OrtValue CreateOutputOrtTensor(TensorElementType elementType, long[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(concatContext);

        return concatContext.Model.InferenceBackend switch
        {
            InferenceBackend.Cpu => OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, outputShape),
            InferenceBackend.Cuda or InferenceBackend.TensorRt => OrtValue.CreateAllocatedTensorValue(cudaAllocator!, elementType, outputShape),
            _ => throw new NotSupportedException($"Batch concat device output is not implemented for backend: {concatContext.Model.InferenceBackend}.")
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
        initializedSingleInputShape = null;
        initializedBatchSize = 0;
        initializedBoundInputCount = 0;
    }

}
