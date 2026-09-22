using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Converts typed YOLO boxes into neutral tracker detections.
/// </summary>
public sealed class Op_DetectionsFromYoloBox : OpBase
{
    readonly string inputKey;
    readonly string outputKey;

    public Op_DetectionsFromYoloBox(string inputKey, string outputKey)
        : base(OpDescriptor.Create(
            "Op_DetectionsFromYoloBox",
            "op.detections.fromYoloBox",
            [VarRequirement.Read<YoloDetectionBatchResult<YoloBox>>(inputKey)],
            [VarRequirement.Write<TrackDetection[]>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
        this.inputKey = inputKey;
        this.outputKey = outputKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        YoloDetectionBatchResult<YoloBox> input = context.Get<YoloDetectionBatchResult<YoloBox>>(inputKey);
        var output = new TrackDetection[input.Detections.Count];

        for(int index = 0; index < output.Length; index++)
        {
            YoloBox detection = input.Detections[index];
            float width = MathF.Abs(detection.W - detection.X);
            float height = MathF.Abs(detection.H - detection.Y);
            output[index] = new TrackDetection(
                MathF.Min(detection.X, detection.W) + (width * 0.5f),
                MathF.Min(detection.Y, detection.H) + (height * 0.5f),
                width,
                height,
                0,
                detection.Class,
                detection.Score,
                index);
        }

        context.Set(outputKey, output);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
