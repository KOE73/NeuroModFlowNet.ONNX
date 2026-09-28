using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Decodes a YOLO box NMS output <c>[batch, items, 6]</c> = (x1, y1, x2, y2, score, class) into typed
/// <see cref="YoloBox"/> arrays, one per request. Field semantics follow the native NMS extractors: <c>X, Y</c> hold the
/// top-left corner and <c>W, H</c> hold the bottom-right corner in model input pixels. Zero-score padding rows are
/// dropped regardless of the threshold.
///
/// RU: Декодирует NMS-выход YOLO box <c>[batch, items, 6]</c> = (x1, y1, x2, y2, score, class) в typed массивы
/// <see cref="YoloBox"/>, по одному на заявку. Семантика полей как у native NMS-экстракторов: <c>X, Y</c> это левый
/// верхний угол, <c>W, H</c> это правый нижний угол в пикселях входа модели. Строки-заполнители с нулевым score
/// отбрасываются независимо от порога.
/// </summary>
public sealed class YoloBoxOrtValueOutputDecoder : IOrtValueBatchOutputDecoder<YoloBox>
{
    const int FieldCount = 6;
    const int X1Index = 0;
    const int Y1Index = 1;
    const int X2Index = 2;
    const int Y2Index = 3;
    const int ScoreIndex = 4;
    const int ClassIndex = 5;

    readonly float scoreThreshold;

    public YoloBoxOrtValueOutputDecoder(float scoreThreshold = 0.5f)
    {
        if(scoreThreshold < 0 || scoreThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(scoreThreshold), "Score threshold must be in [0, 1].");

        this.scoreThreshold = scoreThreshold;
    }

    public IReadOnlyList<YoloBox[]> Decode(
        OrtValue output,
        OnnxModel model,
        string outputName,
        int requestCount,
        IReadOnlyList<int>? requestItemCounts)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(model);

        var outputInfo = output.GetTensorTypeAndShape();
        long[] outputShape = outputInfo.Shape;

        if(outputShape.Length != 3)
            throw new InvalidOperationException($"YOLO box NMS output must be rank 3, actual: [{string.Join(", ", outputShape)}].");

        if(outputShape[2] != FieldCount)
            throw new InvalidOperationException($"YOLO box NMS output field count must be {FieldCount}, actual: {outputShape[2]}.");

        if(outputShape[0] < requestCount)
            throw new InvalidOperationException($"YOLO box NMS output batch {outputShape[0]} is smaller than request count {requestCount}.");

        int itemCount = checked((int)outputShape[1]);

        return outputInfo.ElementDataType switch
        {
            TensorElementType.Float => DecodeFP32(output.GetTensorDataAsSpan<float>(), requestCount, itemCount),
            TensorElementType.Float16 => DecodeFP16(output.GetTensorDataAsSpan<Float16>(), requestCount, itemCount),
            _ => throw new NotSupportedException($"YOLO box NMS output element type is not supported: {outputInfo.ElementDataType}.")
        };
    }

    IReadOnlyList<YoloBox[]> DecodeFP32(ReadOnlySpan<float> data, int requestCount, int itemCount)
    {
        var result = new YoloBox[requestCount][];

        for(int batchIndex = 0; batchIndex < requestCount; batchIndex++)
        {
            ReadOnlySpan<float> batchData = data.Slice(batchIndex * itemCount * FieldCount, itemCount * FieldCount);
            var boxes = new List<YoloBox>(itemCount);

            for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
            {
                ReadOnlySpan<float> row = batchData.Slice(itemIndex * FieldCount, FieldCount);
                if(row[ScoreIndex] > 0f && row[ScoreIndex] >= scoreThreshold)
                    boxes.Add(CreateBox(row[X1Index], row[Y1Index], row[X2Index], row[Y2Index], row[ScoreIndex], row[ClassIndex]));
            }

            result[batchIndex] = boxes.ToArray();
        }

        return result;
    }

    IReadOnlyList<YoloBox[]> DecodeFP16(ReadOnlySpan<Float16> data, int requestCount, int itemCount)
    {
        var result = new YoloBox[requestCount][];

        for(int batchIndex = 0; batchIndex < requestCount; batchIndex++)
        {
            ReadOnlySpan<Float16> batchData = data.Slice(batchIndex * itemCount * FieldCount, itemCount * FieldCount);
            var boxes = new List<YoloBox>(itemCount);

            for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
            {
                ReadOnlySpan<Float16> row = batchData.Slice(itemIndex * FieldCount, FieldCount);
                float score = (float)row[ScoreIndex];
                if(score > 0f && score >= scoreThreshold)
                    boxes.Add(CreateBox((float)row[X1Index], (float)row[Y1Index], (float)row[X2Index], (float)row[Y2Index], score, (float)row[ClassIndex]));
            }

            result[batchIndex] = boxes.ToArray();
        }

        return result;
    }

    static YoloBox CreateBox(float x1, float y1, float x2, float y2, float score, float classId) =>
        new()
        {
            X = x1,
            Y = y1,
            W = x2,
            H = y2,
            Score = score,
            Class = (int)classId
        };
}
