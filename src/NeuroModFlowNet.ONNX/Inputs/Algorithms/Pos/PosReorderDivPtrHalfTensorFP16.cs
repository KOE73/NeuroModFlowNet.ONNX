using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1]: pointer reorder BGR→RGB + normalize (x * 1/255) → ConvertToHalf.
/// Migrated from TranslatorsPositiveNormalized.ToFloat16.Positive_ReorderDivPtr_HalfTensor
/// </summary>
public readonly struct PosReorderDivPtrHalfTensorFP16 : IMatListFillAlgorithm<Float16>
{
    const float ScaleInv = 1f / 255f;

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
                    *pRed = *src++ * ScaleInv;
                    *pGreen = *src++ * ScaleInv;
                    *pBlue = *src++ * ScaleInv;
                }
            }
        }

        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(spanFloat, dst);
    }
}
