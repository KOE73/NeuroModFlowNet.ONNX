using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Symmetric [-1,1] via CvDnn.BlobFromImage → float buffer.
/// </summary>
public readonly struct SymCvdnnFP32 : IMatFillAlgorithm<float>, IMatListFillAlgorithm<float>
{
    const float ShiftSym = 127.5f;
    const float ScaleSymInv = 1f / 127.5f;

    public static unsafe void Fill(Mat image, Span<float> buffer, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImage(image,
            scaleFactor: ScaleSymInv,
            mean: new Scalar(ShiftSym, ShiftSym, ShiftSym),
            swapRB: true, crop: false);
        new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total()).CopyTo(buffer);
    }

    public static unsafe void Fill(List<Mat> mats, Span<float> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImages(mats,
            scaleFactor: ScaleSymInv,
            mean: new Scalar(ShiftSym, ShiftSym, ShiftSym),
            swapRB: true, crop: false);
        new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total()).CopyTo(buffer);
    }
}
