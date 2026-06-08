using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a YOLO pose model and writes the current frame pose detections as a CPU list.
/// </summary>
/// <remarks>
/// The existing runner API returns a batched detection result even for single-frame inference. The VM command deliberately
/// stores only batch item zero, because this pipeline step represents one current camera/debug frame.
/// </remarks>
public sealed class Model_YoloPose : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly OnnxRunnerResource<Mat, IDetectionResult<YoloPose>> resource;

    public Model_YoloPose(string name, string inputKey, string outputKey, OnnxRunnerResource<Mat, IDetectionResult<YoloPose>> resource)
        : base(OpDescriptor.Create(
            name,
            "onnx.yoloPose",
            [VarRequirement.Read<Mat>(inputKey)],
            [VarRequirement.Write<YoloPoseFrameResult>(outputKey)],
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
        IDetectionResult<YoloPose> result = await resource.ExecuteAsync(input, cancellationToken).ConfigureAwait(false);

        try
        {
            // Materialize the detection sequence before disposing the runner result. Some result implementations are
            // lightweight views over extractor-owned buffers.
            context.Set(outputKey, new YoloPoseFrameResult(result.GetBatch(0).ToArray()));
            return OpResult.Continue;
        }
        finally
        {
            (result as IDisposable)?.Dispose();
        }
    }
}
