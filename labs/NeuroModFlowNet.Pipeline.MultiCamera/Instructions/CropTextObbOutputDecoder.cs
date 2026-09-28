using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Decodes a YOLO OBB NMS output <c>[rows, items, 7]</c> for requests that carry several crops each
/// (<c>requestItemCounts</c>), keeping the crop index of every box. The library <c>YoloObbOrtValueOutputDecoder</c>
/// assumes one image per request; this one walks all rows of a request.
///
/// RU: Декодирует NMS-выход YOLO OBB <c>[rows, items, 7]</c> для заявок, несущих несколько вырезок
/// (<c>requestItemCounts</c>), сохраняя индекс вырезки для каждого бокса. Библиотечный
/// <c>YoloObbOrtValueOutputDecoder</c> считает одну картинку на заявку; этот проходит все строки заявки.
/// </summary>
internal sealed class CropTextObbOutputDecoder : IOrtValueBatchOutputDecoder<CropTextRegion>
{
    const int FieldCount = 7;
    readonly float scoreThreshold;

    public CropTextObbOutputDecoder(float scoreThreshold)
    {
        if(scoreThreshold < 0 || scoreThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(scoreThreshold), "Score threshold must be in [0, 1].");

        this.scoreThreshold = scoreThreshold;
    }

    public IReadOnlyList<CropTextRegion[]> Decode(
        OrtValue output,
        OnnxModel model,
        string outputName,
        int requestCount,
        IReadOnlyList<int>? requestItemCounts)
    {
        var outputInfo = output.GetTensorTypeAndShape();
        long[] shape = outputInfo.Shape;
        if(shape.Length != 3 || shape[2] != FieldCount)
            throw new InvalidOperationException($"Text OBB output must be [rows, items, {FieldCount}], actual: [{string.Join(", ", shape)}].");

        int itemCount = checked((int)shape[1]);
        int totalRows = checked((int)shape[0]);
        int[] rowsPerRequest = requestItemCounts?.ToArray() ?? Enumerable.Repeat(1, requestCount).ToArray();
        if(rowsPerRequest.Length != requestCount || rowsPerRequest.Sum() > totalRows)
            throw new InvalidOperationException($"Text OBB output rows {totalRows} do not cover request item counts [{string.Join(", ", rowsPerRequest)}].");

        var result = new CropTextRegion[requestCount][];
        int rowOffset = 0;

        for(int requestIndex = 0; requestIndex < requestCount; requestIndex++)
        {
            var regions = new List<CropTextRegion>();
            for(int cropIndex = 0; cropIndex < rowsPerRequest[requestIndex]; cropIndex++)
            {
                int row = rowOffset + cropIndex;
                int rowStart = row * itemCount * FieldCount;
                if(outputInfo.ElementDataType == TensorElementType.Float)
                    DecodeRow(output.GetTensorDataAsSpan<float>().Slice(rowStart, itemCount * FieldCount), itemCount, cropIndex, regions);
                else if(outputInfo.ElementDataType == TensorElementType.Float16)
                    DecodeRow(output.GetTensorDataAsSpan<Float16>().Slice(rowStart, itemCount * FieldCount), itemCount, cropIndex, regions);
                else
                    throw new NotSupportedException($"Text OBB output element type is not supported: {outputInfo.ElementDataType}.");
            }

            result[requestIndex] = regions.ToArray();
            rowOffset += rowsPerRequest[requestIndex];
        }

        return result;
    }

    void DecodeRow(ReadOnlySpan<float> rowData, int itemCount, int cropIndex, List<CropTextRegion> regions)
    {
        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            ReadOnlySpan<float> item = rowData.Slice(itemIndex * FieldCount, FieldCount);
            if(item[4] > 0f && item[4] >= scoreThreshold)
                regions.Add(new CropTextRegion(cropIndex, CreateBox(item[0], item[1], item[2], item[3], item[4], item[5], item[6])));
        }
    }

    void DecodeRow(ReadOnlySpan<Float16> rowData, int itemCount, int cropIndex, List<CropTextRegion> regions)
    {
        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            ReadOnlySpan<Float16> item = rowData.Slice(itemIndex * FieldCount, FieldCount);
            float score = (float)item[4];
            if(score > 0f && score >= scoreThreshold)
                regions.Add(new CropTextRegion(cropIndex, CreateBox((float)item[0], (float)item[1], (float)item[2], (float)item[3], score, (float)item[5], (float)item[6])));
        }
    }

    static YoloObb CreateBox(float x, float y, float width, float height, float score, float classId, float angle) =>
        new() { X = x, Y = y, W = width, H = height, Score = score, Class = (int)classId, Angle = angle };
}
