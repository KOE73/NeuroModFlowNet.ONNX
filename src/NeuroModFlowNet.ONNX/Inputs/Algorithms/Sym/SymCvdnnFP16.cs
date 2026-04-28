using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Symmetric [-1,1] via CvDnn.BlobFromImage → Float16 buffer (CvDnn → ConvertToHalf).
/// </summary>
public readonly struct SymCvdnnFP16 : IMatFillAlgorithm<Float16>, IMatListFillAlgorithm<Float16>
{
    const float ShiftSym = 127.5f;
    const float ScaleSymInv = 1f / 127.5f;

    public static unsafe void Fill(Mat image, Span<Float16> buffer, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImage(image,
            scaleFactor: ScaleSymInv,
            mean: new Scalar(ShiftSym, ShiftSym, ShiftSym),
            swapRB: true, crop: false);
        var src = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(src, dst);
    }

    public static unsafe void Fill(List<Mat> mats, Span<Float16> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImages(mats,
            scaleFactor: ScaleSymInv,
            mean: new Scalar(ShiftSym, ShiftSym, ShiftSym),
            swapRB: true, crop: false);
        var src = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        Span<Half> dst = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(src, dst);
    }
}
