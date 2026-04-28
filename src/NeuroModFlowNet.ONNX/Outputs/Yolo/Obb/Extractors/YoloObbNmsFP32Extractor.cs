namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO OBB NMS extractor for FP32 (native output).
/// <br/>
/// RU: Экстрактор YOLO OBB NMS для FP32 (нативный выход).
/// </summary>
public class YoloObbNmsFP32Extractor : YoloObbNmsFP32ExtractorBase<IDetectionResult<YoloObb_FP32_XYWHSCA>>
{
    public override IDetectionResult<YoloObb_FP32_XYWHSCA> Extract()
    {
        return GetOutput();
    }

    public static Dictionary<int, YoloObb_FP32_XYWHSCA[]> ExtractFromTensor(
        float[] data,
        int batchCount,
        int itemCount,
        int fieldCount,
        float threshold)
    {
        return ExtractFromSpan(data, batchCount, itemCount, fieldCount, threshold);
    }

    public static Dictionary<int, YoloObb_FP32_XYWHSCA[]> ExtractFromSpan(
        ReadOnlySpan<float> data,
        int batchCount,
        int itemCount,
        int fieldCount,
        float threshold)
    {
        if(fieldCount != 7)
            throw new ArgumentException("YOLO OBB FP32 layout must contain 7 fields.", nameof(fieldCount));

        var allDetections = MemoryMarshal.Cast<float, YoloObb_FP32_XYWHSCA>(data);
        var result = new Dictionary<int, YoloObb_FP32_XYWHSCA[]>(batchCount);

        for(int batch = 0; batch < batchCount; batch++)
        {
            var batchSpan = allDetections.Slice(batch * itemCount, itemCount);
            var batchResult = new List<YoloObb_FP32_XYWHSCA>(itemCount);

            for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= threshold)
                    batchResult.Add(batchSpan[itemIndex]);

            result[batch] = batchResult.ToArray();
        }

        return result;
    }
}
