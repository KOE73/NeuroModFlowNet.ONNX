namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Symmetric [-1,1]: pointer reorder BGR→RGB + normalize ((x - 127.5) / 127.5) → float buffer.
/// Migrated from TranslatorsSymmetric.ToFloat.Symmetric_ReorderNormPtr
/// </summary>
public readonly struct SymReorderNormPtrFP32 : IMatListFillAlgorithm<float>
{
    const float ShiftSym = 127.5f;
    const float ScaleSymInv = 1f / 127.5f;

    public static unsafe void Fill(List<Mat> mats, Span<float> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        for (int i = 0, offset = 0; i < matsCount; i++, offset += sizeOne)
        {
            Mat mat = mats[i];
            Span<float> spanFloatOne = buffer.Slice(offset, sizeOne);
            fixed (float* dstBase = spanFloatOne)
            {
                byte* src = mat.DataPointer;
                float* pBlue = dstBase;
                float* pGreen = dstBase + pixelsCount;
                float* pRed = dstBase + (pixelsCount * 2);
                float* pEnd = pBlue + pixelsCount;

                for (; pBlue < pEnd; pBlue++, pGreen++, pRed++)
                {
                    *pRed = (*src++ - ShiftSym) * ScaleSymInv;
                    *pGreen = (*src++ - ShiftSym) * ScaleSymInv;
                    *pBlue = (*src++ - ShiftSym) * ScaleSymInv;
                }
            }
        }
    }
}
