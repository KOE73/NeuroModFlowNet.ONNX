using NeuroModFlowNet.CV.Tracking;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Adapter <c>YoloObb[]</c> (the <c>Model_OrtValueInference</c> service result) to neutral <c>TrackDetection[]</c>.
/// Library <c>Op_DetectionsFromYoloObb</c> consumes the Mat-based <c>YoloDetectionBatchResult</c>; this lab op covers the
/// OrtValue endpoint path and optionally keeps a single class, as the old CGP pipeline tracked only "BagTop".
/// <c>SourceIndex</c> always points into the unfiltered input array so downstream can re-associate OCR/overlay data.
///
/// RU: Адаптер <c>YoloObb[]</c> (результат сервиса <c>Model_OrtValueInference</c>) в нейтральные <c>TrackDetection[]</c>.
/// Библиотечный <c>Op_DetectionsFromYoloObb</c> читает Mat-путь <c>YoloDetectionBatchResult</c>; эта lab-инструкция
/// закрывает OrtValue-путь и опционально оставляет один класс, как старый CGP-пайплайн трекал только "BagTop".
/// <c>SourceIndex</c> всегда указывает в исходный нефильтрованный массив.
/// </summary>
internal sealed class Op_DetectionsFromYoloObbArray : OpBase
{
    readonly string inputKey;
    readonly string outputKey;
    readonly int? classId;

    public Op_DetectionsFromYoloObbArray(string inputKey, string outputKey, int? classId = null)
        : base(OpDescriptor.Create(
            "Op_DetectionsFromYoloObbArray",
            "op.detections.fromYoloObbArray",
            [VarRequirement.Read<YoloObb[]>(inputKey)],
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

        YoloObb[] input = context.Get<YoloObb[]>(inputKey);
        var output = new List<TrackDetection>(input.Length);

        for(int index = 0; index < input.Length; index++)
        {
            ref readonly YoloObb detection = ref input[index];
            if(classId is { } requiredClass && detection.Class != requiredClass)
                continue;

            output.Add(new TrackDetection(
                detection.X,
                detection.Y,
                detection.W,
                detection.H,
                detection.Angle,
                detection.Class,
                detection.Score,
                index));
        }

        context.Set(outputKey, output.ToArray());
        return ValueTask.FromResult(OpResult.Continue);
    }
}
