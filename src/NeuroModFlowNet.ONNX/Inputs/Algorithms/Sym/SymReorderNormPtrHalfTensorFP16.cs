using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Symmetric [-1,1]: pointer reorder + normalize → float → ConvertToHalf.
/// Migrated from TranslatorsSymmetric.ToFloat16.Symmetric_ReorderNormPtr_HalfTensor
/// </summary>
public readonly struct SymReorderNormPtrHalfTensorFP16 : IMatListFillAlgorithm<Float16>
{
    const float ShiftSym = 127.5f;
    const float ScaleSymInv = 1f / 127.5f;

    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        int sizeAll = sizeOne * matsCount;

        using var ownerFloat = MemoryPool<float>.Shared.Rent(sizeAll);
        Span<float> spanFloat = ownerFloat.Memory.Span.Slice(0, sizeAll);

        for (int i = 0, offset = 0; i < matsCount; i++, offset += sizeOne)
        {
            Mat mat = mats[i];
            Span<float> spanFloatOne = spanFloat.Slice(offset, sizeOne);
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

        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(spanFloat, dst);
    }
}
