using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class YoloObbOrtValueOutputDecoder : IOrtValueBatchOutputDecoder<YoloObb>
{
    const int FieldCount = 7;
    const int XIndex = 0;
    const int YIndex = 1;
    const int WIndex = 2;
    const int HIndex = 3;
    const int ScoreIndex = 4;
    const int ClassIndex = 5;
    const int AngleIndex = 6;

    readonly float scoreThreshold;

    public YoloObbOrtValueOutputDecoder(float scoreThreshold = 0.5f)
    {
        if(scoreThreshold < 0 || scoreThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(scoreThreshold), "Score threshold must be in [0, 1].");

        this.scoreThreshold = scoreThreshold;
    }

    public IReadOnlyList<YoloObb[]> Decode(
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
            throw new InvalidOperationException($"YOLO OBB output must be rank 3, actual: [{string.Join(", ", outputShape)}].");

        if(outputShape[2] != FieldCount)
            throw new InvalidOperationException($"YOLO OBB output field count must be {FieldCount}, actual: {outputShape[2]}.");

        if(outputShape[0] < requestCount)
            throw new InvalidOperationException($"YOLO OBB output batch {outputShape[0]} is smaller than request count {requestCount}.");

        int itemCount = checked((int)outputShape[1]);

        return outputInfo.ElementDataType switch
        {
            TensorElementType.Float => DecodeFP32(output.GetTensorDataAsSpan<float>(), requestCount, itemCount),
            TensorElementType.Float16 => DecodeFP16(output.GetTensorDataAsSpan<Half>(), requestCount, itemCount),
            _ => throw new NotSupportedException($"YOLO OBB output element type is not supported: {outputInfo.ElementDataType}.")
        };
    }

    IReadOnlyList<YoloObb[]> DecodeFP32(ReadOnlySpan<float> data, int requestCount, int itemCount)
    {
        var result = new YoloObb[requestCount][];

        for(int batchIndex = 0; batchIndex < requestCount; batchIndex++)
            result[batchIndex] = DecodeBatchFP32(data.Slice(batchIndex * itemCount * FieldCount, itemCount * FieldCount), itemCount);

        return result;
    }

    IReadOnlyList<YoloObb[]> DecodeFP16(ReadOnlySpan<Half> data, int requestCount, int itemCount)
    {
        var result = new YoloObb[requestCount][];

        for(int batchIndex = 0; batchIndex < requestCount; batchIndex++)
            result[batchIndex] = DecodeBatchFP16(data.Slice(batchIndex * itemCount * FieldCount, itemCount * FieldCount), itemCount);

        return result;
    }

    YoloObb[] DecodeBatchFP32(ReadOnlySpan<float> batchData, int itemCount)
    {
        var boxes = new List<YoloObb>(itemCount);

        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            ReadOnlySpan<float> row = batchData.Slice(itemIndex * FieldCount, FieldCount);
            if(row[ScoreIndex] >= scoreThreshold)
                boxes.Add(CreateBox(row[XIndex], row[YIndex], row[WIndex], row[HIndex], row[ScoreIndex], row[ClassIndex], row[AngleIndex]));
        }

        return boxes.ToArray();
    }

    YoloObb[] DecodeBatchFP16(ReadOnlySpan<Half> batchData, int itemCount)
    {
        var boxes = new List<YoloObb>(itemCount);

        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            ReadOnlySpan<Half> row = batchData.Slice(itemIndex * FieldCount, FieldCount);
            float score = (float)row[ScoreIndex];
            if(score >= scoreThreshold)
                boxes.Add(CreateBox((float)row[XIndex], (float)row[YIndex], (float)row[WIndex], (float)row[HIndex], score, (float)row[ClassIndex], (float)row[AngleIndex]));
        }

        return boxes.ToArray();
    }

    static YoloObb CreateBox(float x, float y, float width, float height, float score, float classId, float angle) =>
        new()
        {
            X = x,
            Y = y,
            W = width,
            H = height,
            Score = score,
            Class = (int)classId,
            Angle = angle
        };
}
