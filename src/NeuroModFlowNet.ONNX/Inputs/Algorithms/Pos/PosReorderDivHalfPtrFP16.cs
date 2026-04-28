using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1]: pointer reorder + normalize + direct Half write (no intermediate float buffer).
/// Migrated from TranslatorsPositiveNormalized.ToFloat16.Positive_ReorderDivHalfPtr
/// </summary>
public readonly struct PosReorderDivHalfPtrFP16 : IMatListFillAlgorithm<Float16>
{
    const float ScaleInv = 1f / 255f;

    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        for(int i = 0, offset = 0; i < matsCount; i++, offset += sizeOne)
        {
            Mat mat = mats[i];
            Span<Float16> spanOne = buffer.Slice(offset, sizeOne);

            byte* src = mat.DataPointer;
            Half* dstBase = (Half*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(spanOne));
            Half* pBlue = dstBase;
            Half* pGreen = dstBase + pixelsCount;
            Half* pRed = dstBase + (pixelsCount * 2);
            Half* pEnd = pBlue + pixelsCount;

            for(; pBlue < pEnd; pBlue++, pGreen++, pRed++)
            {
                *pRed = (Half)(*src++ * ScaleInv);
                *pGreen = (Half)(*src++ * ScaleInv);
                *pBlue = (Half)(*src++ * ScaleInv);
            }
        }
    }
}
