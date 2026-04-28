namespace NeuroModFlowNet.ONNX;

public class YoloBoxResultsFP16 : IDetectionResult<YoloBox>
{
    readonly Dictionary<int, YoloBox_FP16_XYWHSC[]> _dict;

    public YoloBoxResultsFP16(Dictionary<int, YoloBox_FP16_XYWHSC[]> dict) => _dict = dict;

    public int BatchCount => _dict.Count;

    public int GetResultCount(int batchIndex = 0) => _dict.TryGetValue(batchIndex, out var arr) ? arr.Length : 0;

    public YoloBox GetResult(int index, int batchIndex = 0)
    {
        // Берем оригинальную структуру FP16 (работа по ссылке массива)
        ref var raw = ref _dict[batchIndex][index];

        // Возвращаем новую структуру. 
        // ВНИМАНИЕ: Она создается на Стеке (Stack), а не в Куче (Heap)!
        // GC ее даже не увидит.
        return new YoloBox
        {
            X = (float)raw.X,
            Y = (float)raw.Y,
            W = (float)raw.W,
            H = (float)raw.H,
            Score = (float)raw.Score,
            Class = (int)raw.Class
        };
    }

    public ReadOnlySpan<YoloBox> GetBatch(int batchIndex = 0)
    {
        if(!_dict.TryGetValue(batchIndex, out var arr)) return [];
        var result = new YoloBox[arr.Length];
        for(int i = 0; i < arr.Length; i++)
            result[i] = GetResult(i, batchIndex);
        return result;
    }
}
