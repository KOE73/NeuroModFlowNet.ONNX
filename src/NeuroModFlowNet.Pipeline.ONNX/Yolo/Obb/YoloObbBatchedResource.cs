using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Batchable YOLO OBB resource backed by the existing NeuroModFlowNet.ONNX runner API.
/// </summary>
public sealed class YoloObbBatchedResource :
    OnnxBatchedResourceBase<Mat, YoloDetectionBatchResult<YoloObb>>
{
    readonly IRunner<List<Mat>, IDetectionResult<YoloObb>> runner;

    public YoloObbBatchedResource(
        string name,
        IRunner<List<Mat>, IDetectionResult<YoloObb>> runner,
        OnnxBatchedResourceOptions options)
        : base(name, options)
    {
        this.runner = runner ?? throw new ArgumentNullException(nameof(runner));
    }

    public static YoloObbBatchedResource Create(YoloObbResourceDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.ModelPath);

        var context = new OnnxExecutionContext(new OnnxModel(definition.ModelPath), ownsModel: true);
        IRunner<List<Mat>, IDetectionResult<YoloObb>> runner = definition.RunnerKind switch
        {
            YoloObbRunnerKind.ListPosCvdnnFP32 => YoloObbFactory.List_PosCvdnn_FP32(context),
            YoloObbRunnerKind.ListSymCvdnnFP32 => YoloObbFactory.List_SymCvdnn_FP32(context),
            _ => throw new NotSupportedException($"Unsupported YOLO OBB runner kind: {definition.RunnerKind}.")
        };

        return new YoloObbBatchedResource(definition.Name, runner, definition.Batching);
    }

    protected override void DisposeCore() => runner.Dispose();

    protected override ValueTask<IReadOnlyList<YoloDetectionBatchResult<YoloObb>>> ExecuteBatchAsync(
        IReadOnlyList<Mat> inputs,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        List<Mat> paddedInputs = CreatePaddedInputs(inputs);
        IDetectionResult<YoloObb> result = runner.Predict(paddedInputs);

        try
        {
            YoloDetectionBatchResult<YoloObb>[] outputs = new YoloDetectionBatchResult<YoloObb>[inputs.Count];

            // Only real requests are copied back. Padded images exist solely to satisfy fixed-batch ONNX input shape and
            // their detections must never become visible as pipeline transaction values.
            for(int batchIndex = 0; batchIndex < inputs.Count; batchIndex++)
                outputs[batchIndex] = new YoloDetectionBatchResult<YoloObb>(result.GetBatch(batchIndex).ToArray());

            return ValueTask.FromResult<IReadOnlyList<YoloDetectionBatchResult<YoloObb>>>(outputs);
        }
        finally
        {
            (result as IDisposable)?.Dispose();
            DisposePadding(inputs.Count, paddedInputs);
        }
    }

    List<Mat> CreatePaddedInputs(IReadOnlyList<Mat> inputs)
    {
        var paddedInputs = new List<Mat>(Options.MaxBatchSize);
        paddedInputs.AddRange(inputs);

        if(inputs.Count == 0 || inputs.Count >= Options.MaxBatchSize)
            return paddedInputs;

        Mat template = inputs[0];
        while(paddedInputs.Count < Options.MaxBatchSize)
        {
            // Batch-4 models need exactly four inputs. Padding is internal to the resource; dummy results are discarded
            // and never become transaction values.
            paddedInputs.Add(new Mat(template.Rows, template.Cols, template.Type(), Scalar.Black));
        }

        return paddedInputs;
    }

    static void DisposePadding(int realInputCount, List<Mat> paddedInputs)
    {
        for(int index = realInputCount; index < paddedInputs.Count; index++)
            paddedInputs[index].Dispose();
    }
}
