using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a YOLO classification model and writes the compact classification result for the current frame.
/// </summary>
/// <remarks>
/// Classification shares the generic <see cref="IBatchedResult"/> runner surface with other model families. This command
/// performs the type boundary check so later VM steps can consume a strongly named pipeline result.
/// </remarks>
public sealed class Model_YoloCls : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly OnnxRunnerResource<Mat, IBatchedResult> resource;

    public Model_YoloCls(string name, string inputKey, string outputKey, OnnxRunnerResource<Mat, IBatchedResult> resource)
        : base(OpDescriptor.Create(
            name,
            "onnx.yoloCls",
            [VarRequirement.Read<Mat>(inputKey)],
            [VarRequirement.Write<YoloClassificationFrameResult>(outputKey)],
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

        if(result is not YoloCls classification)
            throw new NotSupportedException($"Unsupported YOLO Cls result type: {result.GetType().Name}");

        context.Set(outputKey, YoloClassificationFrameResult.From(classification));
        return OpResult.Continue;
    }
}
