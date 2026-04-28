using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1] via CvDnn.BlobFromImage → Float16 buffer (CvDnn float → ConvertToHalf).
/// </summary>
public readonly struct PosCvdnnFP16 : IMatFillAlgorithm<Float16>, IMatListFillAlgorithm<Float16>
{
    public static unsafe void Fill(Mat image, Span<Float16> buffer, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImage(image, 1.0 / 255.0, swapRB: true, crop: false);
        var src = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(src, dst);
    }

    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImages(mats, 1.0 / 255.0, swapRB: true, crop: false);
        var src = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(src, dst);
    }
}
