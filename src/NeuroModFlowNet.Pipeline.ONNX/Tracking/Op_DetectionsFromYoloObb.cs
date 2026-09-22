using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Converts typed YOLO OBB detections into neutral tracker detections.
/// </summary>
public sealed class Op_DetectionsFromYoloObb : OpBase
{
    readonly string inputKey;
    readonly string outputKey;

    public Op_DetectionsFromYoloObb(string inputKey, string outputKey)
        : base(OpDescriptor.Create(
            "Op_DetectionsFromYoloObb",
            "op.detections.fromYoloObb",
            [VarRequirement.Read<YoloDetectionBatchResult<YoloObb>>(inputKey)],
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

        YoloDetectionBatchResult<YoloObb> input = context.Get<YoloDetectionBatchResult<YoloObb>>(inputKey);
        var output = new TrackDetection[input.Detections.Count];

        for(int index = 0; index < output.Length; index++)
        {
            YoloObb detection = input.Detections[index];
            output[index] = new TrackDetection(
                detection.X,
                detection.Y,
                detection.W,
                detection.H,
                detection.Angle,
                detection.Class,
                detection.Score,
                index);
        }

        context.Set(outputKey, output);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
