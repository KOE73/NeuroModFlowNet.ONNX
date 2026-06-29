using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class OrtValueBatchedInferenceEndpoint<TOutput> :
    OnnxBatchedResourceBase<OrtValue, TOutput[]>
{
    readonly string modelPath;
    readonly InferenceBackend executionBackend;
    readonly IOrtValueBatchInputAssembler batchInputAssembler;
    readonly IOrtValueOutputShapeResolver outputShapeResolver;
    readonly IOrtValueBatchOutputDecoder<TOutput> outputDecoder;
    readonly Action<ExecutionProviderConfig>? configureExecutionProvider;
    OnnxExecutionContext? modelContext;

    public OrtValueBatchedInferenceEndpoint(
        string name,
        string modelPath,
        InferenceBackend executionBackend,
        OnnxBatchedResourceOptions options,
        IOrtValueBatchInputAssembler batchInputAssembler,
        IOrtValueOutputShapeResolver outputShapeResolver,
        IOrtValueBatchOutputDecoder<TOutput> outputDecoder,
        Action<ExecutionProviderConfig>? configureExecutionProvider = null)
        : base(name, options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        this.modelPath = modelPath;
        this.executionBackend = executionBackend;
        this.batchInputAssembler = batchInputAssembler ?? throw new ArgumentNullException(nameof(batchInputAssembler));
        this.outputShapeResolver = outputShapeResolver ?? throw new ArgumentNullException(nameof(outputShapeResolver));
        this.outputDecoder = outputDecoder ?? throw new ArgumentNullException(nameof(outputDecoder));
        this.configureExecutionProvider = configureExecutionProvider;
    }

    protected override ValueTask<IReadOnlyList<TOutput[]>> ExecuteBatchAsync(
        IReadOnlyList<OrtValue> inputs,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        EnsureContext();

        using OrtValueBatchInput modelInput = batchInputAssembler.Assemble(inputs, executionBackend, cancellationToken);
        using OrtValue modelOutput = CreateOutputOrtTensor(modelInput.BatchSize);

        RunWithFreshBinding(modelInput.Value, modelOutput);

        IReadOnlyList<TOutput[]> outputs = outputDecoder.Decode(
            modelOutput,
            modelContext!.Model,
            modelContext.Model.PrimaryOutputName,
            inputs.Count);

        if(outputs.Count != inputs.Count)
            throw new InvalidOperationException($"Decoder returned {outputs.Count} outputs for {inputs.Count} requests.");

        return ValueTask.FromResult(outputs);
    }

    void EnsureContext()
    {
        if(modelContext is not null)
            return;

        modelContext = new OnnxExecutionContext(new OnnxModel(modelPath, executionBackend, configureExecutionProvider), ownsModel: true);
    }

    OrtValue CreateOutputOrtTensor(int requestCount)
    {
        ArgumentNullException.ThrowIfNull(modelContext);

        string outputName = modelContext.Model.PrimaryOutputName;
        TensorElementType outputElementType = modelContext.Model.GetOutputElementType(outputName);
        long[] outputShape = outputShapeResolver.ResolveOutputShape(modelContext.Model, outputName, requestCount);

        return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, outputElementType, outputShape);
    }

    void RunWithFreshBinding(OrtValue inputOrtValue, OrtValue outputOrtValue)
    {
        ArgumentNullException.ThrowIfNull(modelContext);

        modelContext.IoBinding.ClearBoundInputs();
        modelContext.IoBinding.ClearBoundOutputs();

        try
        {
            modelContext.IoBinding.BindInput(modelContext.Model.PrimaryInputName, inputOrtValue);
            modelContext.IoBinding.BindOutput(modelContext.Model.PrimaryOutputName, outputOrtValue);
            modelContext.Model.Session.RunWithBinding(modelContext.RunOptions, modelContext.IoBinding);
            modelContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            modelContext.IoBinding.ClearBoundInputs();
            modelContext.IoBinding.ClearBoundOutputs();
        }
    }

    protected override void DisposeCore()
    {
        modelContext?.Dispose();
        batchInputAssembler.Dispose();
    }
}
