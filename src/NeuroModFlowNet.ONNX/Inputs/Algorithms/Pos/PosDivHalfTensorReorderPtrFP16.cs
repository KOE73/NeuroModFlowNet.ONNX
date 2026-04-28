using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1]: byte→float (TensorPrimitives) → Multiply(1/255) → ConvertToHalf → Reorder HWC→CHW (pointer).
/// Migrated from TranslatorsPositiveNormalized.ToFloat16.Positive_DivHalfTensor_ReorderPtr
/// </summary>
public readonly struct PosDivHalfTensorReorderPtrFP16 : IMatListFillAlgorithm<Float16>
{
    const float ScaleInv = 1f / 255f;

    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        int sizeAll = sizeOne * matsCount;

        using var ownerFloat = MemoryPool<float>.Shared.Rent(sizeAll);
        Span<float> spanFloat = ownerFloat.Memory.Span.Slice(0, sizeAll);

        for(int i = 0, offset = 0; i < matsCount; i++, offset += sizeOne)
            TensorPrimitives.ConvertTruncating(
                new ReadOnlySpan<byte>(mats[i].DataPointer, sizeOne),
                spanFloat.Slice(offset, sizeOne));

        TensorPrimitives.Multiply(spanFloat, ScaleInv, spanFloat);

        using var ownerHalf = MemoryPool<Half>.Shared.Rent(sizeAll);
        Span<Half> spanHalf = ownerHalf.Memory.Span.Slice(0, sizeAll);
        TensorPrimitives.ConvertToHalf(spanFloat, spanHalf);

        // HWC → CHW reorder
        Span<short> viewIn = MemoryMarshal.Cast<Half, short>(spanHalf);
        Span<short> viewOut = MemoryMarshal.Cast<Float16, short>(buffer);

        int sizeOneImage = pixelsCount * 3;
        fixed(short* srcBase = viewIn, dstBase = viewOut)
        {
            for(int m = 0; m < matsCount; m++)
            {
                short* src = srcBase + (m * sizeOneImage);
                short* dstBaseOne = dstBase + (m * sizeOneImage);
                short* pBlue = dstBaseOne;
                short* pGreen = dstBaseOne + pixelsCount;
                short* pRed = dstBaseOne + (pixelsCount * 2);
                short* pEnd = pBlue + pixelsCount;

                for(; pBlue < pEnd; pBlue++, pGreen++, pRed++)
                {
                    *pRed = *src++;
                    *pGreen = *src++;
                    *pBlue = *src++;
                }
            }
        }
    }
}
