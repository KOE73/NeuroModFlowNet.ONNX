using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Batchable YOLO Box resource backed by the existing NeuroModFlowNet.ONNX runner API.
/// </summary>
public sealed class YoloBoxBatchedResource :
    OnnxBatchedResourceBase<Mat, YoloDetectionBatchResult<YoloBox>>
{
    readonly IRunner<List<Mat>, IDetectionResult<YoloBox>> runner;

    public YoloBoxBatchedResource(
        string name,
        IRunner<List<Mat>, IDetectionResult<YoloBox>> runner,
        OnnxBatchedResourceOptions options)
        : base(name, options)
    {
        this.runner = runner ?? throw new ArgumentNullException(nameof(runner));
    }

    public static YoloBoxBatchedResource Create(YoloBoxResourceDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.ModelPath);

        var context = new OnnxExecutionContext(new OnnxModel(definition.ModelPath), ownsModel: true);
        IRunner<List<Mat>, IDetectionResult<YoloBox>> runner = definition.RunnerKind switch
        {
            YoloBoxRunnerKind.ListPosCvdnnFP32 => YoloBoxFactory.List_PosCvdnn_FP32(context),
            YoloBoxRunnerKind.ListSymCvdnnFP32 => YoloBoxFactory.List_SymCvdnn_FP32(context),
            YoloBoxRunnerKind.ListPosCvdnnFP16 => YoloBoxFactory.List_PosCvdnn_FP16(context),
            YoloBoxRunnerKind.ListSymCvdnnFP16 => YoloBoxFactory.List_SymCvdnn_FP16(context),
            _ => throw new NotSupportedException($"Unsupported YOLO Box runner kind: {definition.RunnerKind}.")
        };

        return new YoloBoxBatchedResource(definition.Name, runner, definition.Batching);
    }

    protected override void DisposeCore() => runner.Dispose();

    protected override ValueTask<IReadOnlyList<YoloDetectionBatchResult<YoloBox>>> ExecuteBatchAsync(
        IReadOnlyList<Mat> inputs,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        IDetectionResult<YoloBox> result = runner.Predict(inputs.ToList());

        try
        {
            YoloDetectionBatchResult<YoloBox>[] outputs = new YoloDetectionBatchResult<YoloBox>[inputs.Count];

            // The VM waits for one result per submitted request. Copy each batch item out of the runner result before
            // disposing it so no transaction keeps references to extractor-owned buffers.
            for(int batchIndex = 0; batchIndex < inputs.Count; batchIndex++)
                outputs[batchIndex] = new YoloDetectionBatchResult<YoloBox>(result.GetBatch(batchIndex).ToArray());

            return ValueTask.FromResult<IReadOnlyList<YoloDetectionBatchResult<YoloBox>>>(outputs);
        }
        finally
        {
            (result as IDisposable)?.Dispose();
        }
    }
}
