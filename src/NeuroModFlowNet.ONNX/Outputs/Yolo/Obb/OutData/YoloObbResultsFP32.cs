
namespace NeuroModFlowNet.ONNX;

public class YoloObbResultsFP32 : IDetectionResult<YoloObb>
{
    readonly Dictionary<int, YoloObb_FP32_XYWHSCA[]> _dict;

    public YoloObbResultsFP32(Dictionary<int, YoloObb_FP32_XYWHSCA[]> dict) => _dict = dict;

    public int BatchCount => _dict.Count;

    public int GetResultCount(int batchIndex = 0) => _dict.TryGetValue(batchIndex, out var arr) ? arr.Length : 0;

    public YoloObb GetResult(int index, int batchIndex = 0)
    {
        // Берем оригинальную структуру FP32 (работа по ссылке массива)
        ref var raw = ref _dict[batchIndex][index];

        // Возвращаем новую структуру. 
        return raw.AsStd();
    }

    public ReadOnlySpan<YoloObb> GetBatch(int batchIndex = 0)
    {
        if(!_dict.TryGetValue(batchIndex, out var arr)) return [];
        var result = new YoloObb[arr.Length];
        for(int i = 0; i < arr.Length; i++)
            result[i] = GetResult(i, batchIndex);
        return result;
    }
}
