using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1]: pointer reorder BGR→RGB (no normalize) → Multiply(1/255) → ConvertToHalf.
/// Migrated from TranslatorsPositiveNormalized.ToFloat16.Positive_ReorderPtr_DivHalfTensor
/// </summary>
public readonly struct PosReorderPtrDivHalfTensorFP16 : IMatListFillAlgorithm<Float16>
{
    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        int sizeAll = sizeOne * matsCount;

        using var ownerFloat = MemoryPool<float>.Shared.Rent(sizeAll);
        Span<float> spanFloat = ownerFloat.Memory.Span.Slice(0, sizeAll);

        for(int i = 0, offset = 0; i < matsCount; i++, offset += sizeOne)
        {
            Mat mat = mats[i];
            Span<float> spanFloatOne = spanFloat.Slice(offset, sizeOne);
            fixed(float* dstBase = spanFloatOne)
            {
                byte* src = mat.DataPointer;
                float* pBlue = dstBase;
                float* pGreen = dstBase + pixelsCount;
                float* pRed = dstBase + (pixelsCount * 2);
                float* pEnd = pBlue + pixelsCount;

                for(; pBlue < pEnd; pBlue++, pGreen++, pRed++)
                {
                    *pRed = *src++;
                    *pGreen = *src++;
                    *pBlue = *src++;
                }
            }
        }

        TensorPrimitives.Multiply(spanFloat, 1f / 255f, spanFloat);

        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(spanFloat, dst);
    }
}
