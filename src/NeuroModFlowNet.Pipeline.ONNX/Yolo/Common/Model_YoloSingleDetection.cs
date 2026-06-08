using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a single-image detection runner and writes a compact CPU detection list.
/// </summary>
/// <remarks>
/// This is the bridge for current batch-1 demo assets. Batched resources stay available for true batch-capable models,
/// while this instruction keeps the MVP compatible with the existing single-frame factories.
/// </remarks>
public sealed class Model_YoloSingleDetection<TDetection> : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly OnnxRunnerResource<Mat, IDetectionResult<TDetection>> resource;

    public Model_YoloSingleDetection(
        string name,
        string operation,
        string inputKey,
        string outputKey,
        OnnxRunnerResource<Mat, IDetectionResult<TDetection>> resource)
        : base(OpDescriptor.Create(
            name,
            operation,
            [VarRequirement.Read<Mat>(inputKey)],
            [VarRequirement.Write<YoloDetectionBatchResult<TDetection>>(outputKey)],
            hasSideEffects: true))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.resource = resource ?? throw new ArgumentNullException(nameof(resource));
    }

    public override async ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        Mat input = context.Get<Mat>(inputKey);
        IDetectionResult<TDetection> result = await resource.ExecuteAsync(input, cancellationToken).ConfigureAwait(false);

        try
        {
            // Copy the current batch item out before disposing the detection result. This keeps native extractor buffers
            // and VM transaction values from sharing lifetime accidentally.
            context.Set(outputKey, new YoloDetectionBatchResult<TDetection>(result.GetBatch(0).ToArray()));
            return OpResult.Continue;
        }
        finally
        {
            (result as IDisposable)?.Dispose();
        }
    }
}
