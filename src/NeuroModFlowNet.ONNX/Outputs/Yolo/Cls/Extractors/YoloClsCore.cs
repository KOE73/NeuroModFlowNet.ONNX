namespace NeuroModFlowNet.ONNX;     

public static class YoloClsCore
{
    public static (int ClassId, float Score) GetBestForBatch(
        ReadOnlySpan<float> data,
        int batch,
        int classesCount)
    {
        Debug.Assert(batch >= 0);
        Debug.Assert(classesCount > 0);

        int start = batch * classesCount;
        Debug.Assert(start >= 0);
        Debug.Assert(start + classesCount <= data.Length);

        var batchSpan = data.Slice(start, classesCount);

        int bestIndex = TensorPrimitives.IndexOfMax(batchSpan);
        return (bestIndex, batchSpan[bestIndex]);
    }


    public static YoloClsTopKItem_FP32[] GetTopKForBatch(
       ReadOnlySpan<float> data,
       int batch,
       int classesCount,
       int k)
    {
        Debug.Assert(batch >= 0);
        Debug.Assert(classesCount > 0);
        Debug.Assert(k > 0);
        Debug.Assert(k <= classesCount);

        int start = batch * classesCount;
        Debug.Assert(start >= 0);
        Debug.Assert(start + classesCount <= data.Length);

        var batchSpan = data.Slice(start, classesCount);

        if(k <= 32)
        {
            Span<YoloClsTopKItem_FP32> top = stackalloc YoloClsTopKItem_FP32[k];
            Init(top);
            FillTopK(batchSpan, top);
            return top.ToArray();
        }

        var result = new YoloClsTopKItem_FP32[k];
        Init(result);
        FillTopK(batchSpan, result);
        return result;
    }

    private static void Init(Span<YoloClsTopKItem_FP32> top)
    {
        for(int itemIndex = 0; itemIndex < top.Length; itemIndex++)
            top[itemIndex] = new YoloClsTopKItem_FP32(-1, float.NegativeInfinity);
    }

    private static void FillTopK(
        ReadOnlySpan<float> batchSpan,
        Span<YoloClsTopKItem_FP32> top)
    {
        for(int classId = 0; classId < batchSpan.Length; classId++)
        {
            float score = batchSpan[classId];

            if(score <= top[^1].Score)
                continue;

            int pos = top.Length - 1;
            while(pos > 0 && score > top[pos - 1].Score)
            {
                top[pos] = top[pos - 1];
                pos--;
            }

            top[pos] = new YoloClsTopKItem_FP32(classId, score);
        }
    }
}