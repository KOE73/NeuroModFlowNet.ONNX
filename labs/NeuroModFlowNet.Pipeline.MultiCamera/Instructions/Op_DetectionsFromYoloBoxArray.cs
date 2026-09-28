using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Adapter <c>YoloBox[]</c> (OrtValue endpoint result, corners in <c>X, Y</c> / <c>W, H</c>) to neutral
/// <c>TrackDetection[]</c> (center, size, angle 0). Optionally keeps a single class, as the old CGP pipeline tracked only
/// "BagTop". <c>SourceIndex</c> points into the unfiltered input array.
///
/// RU: Адаптер <c>YoloBox[]</c> (результат OrtValue endpoint, углы в <c>X, Y</c> / <c>W, H</c>) в нейтральные
/// <c>TrackDetection[]</c> (центр, размер, угол 0). Опционально оставляет один класс, как старый CGP-пайплайн трекал
/// только "BagTop". <c>SourceIndex</c> указывает в исходный нефильтрованный массив.
/// </summary>
internal sealed class Op_DetectionsFromYoloBoxArray : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly int? classId;

    public Op_DetectionsFromYoloBoxArray(string inputKey, string outputKey, int? classId = null)
        : base(OpDescriptor.Create(
            "Op_DetectionsFromYoloBoxArray",
            "op.detections.fromYoloBoxArray",
            [VarRequirement.Read<YoloBox[]>(inputKey)],
            [VarRequirement.Write<TrackDetection[]>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);
        this.inputKey = inputKey;
        this.outputKey = outputKey;
        this.classId = classId;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        YoloBox[] input = context.Get<YoloBox[]>(inputKey);
        var output = new List<TrackDetection>(input.Length);

        for(int index = 0; index < input.Length; index++)
        {
            ref readonly YoloBox detection = ref input[index];
            if(classId is { } requiredClass && detection.Class != requiredClass)
                continue;

            float width = MathF.Abs(detection.W - detection.X);
            float height = MathF.Abs(detection.H - detection.Y);
            output.Add(new TrackDetection(
                MathF.Min(detection.X, detection.W) + (width * 0.5f),
                MathF.Min(detection.Y, detection.H) + (height * 0.5f),
                width,
                height,
                0f,
                detection.Class,
                detection.Score,
                index));
        }

        context.Set(outputKey, output.ToArray());
        return ValueTask.FromResult(OpResult.Continue);
    }
}
