using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

/// <summary>
/// Positive [0,1] via CvDnn.BlobFromImage → float buffer.
/// </summary>
public readonly struct PosCvdnnFP32 : IMatFillAlgorithm<float>, IMatListFillAlgorithm<float>
{
    public static unsafe void Fill(Mat image, Span<float> buffer, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImage(image, 1.0 / 255.0, swapRB: true, crop: false);
        new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total()).CopyTo(buffer);
    }

    public static unsafe void Fill(List<Mat> mats, Span<float> buffer,
        int matsCount, int sizeOne, int pixelsCount)
    {
        using var blob = CvDnn.BlobFromImages(mats, 1.0 / 255.0, swapRB: true, crop: false);
        new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total()).CopyTo(buffer);
    }
}
