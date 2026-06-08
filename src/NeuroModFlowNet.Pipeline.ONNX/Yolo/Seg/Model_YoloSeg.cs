using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a YOLO segmentation model and writes a frame-level result wrapper with the supported mask tensor layout.
/// </summary>
/// <remarks>
/// Current segmentation runners expose their output through <see cref="IBatchedResult"/>. This command is the narrow
/// place where the generic runner result becomes the pipeline-specific <see cref="YoloSegFrameResult"/> contract.
/// </remarks>
public sealed class Model_YoloSeg : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly OnnxRunnerResource<Mat, IBatchedResult> resource;

    public Model_YoloSeg(string name, string inputKey, string outputKey, OnnxRunnerResource<Mat, IBatchedResult> resource)
        : base(OpDescriptor.Create(
            name,
            "onnx.yoloSeg",
            [VarRequirement.Read<Mat>(inputKey)],
            [VarRequirement.Write<YoloSegFrameResult>(outputKey)],
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
        IBatchedResult result = await resource.ExecuteAsync(input, cancellationToken).ConfigureAwait(false);

        YoloSegFrameResult frameResult = result switch
        {
            YoloSegResult_FP32_Mask32 fp32 => new YoloSegFrameResult(fp32),
            YoloSegResult_FP16_Mask32 fp16 => new YoloSegFrameResult(fp16),
            _ => throw new NotSupportedException($"Unsupported YOLO Seg result type: {result.GetType().Name}")
        };

        context.Set(outputKey, frameResult);
        return OpResult.Continue;
    }
}
