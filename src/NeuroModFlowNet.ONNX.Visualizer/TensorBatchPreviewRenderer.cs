using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Numbered grid of an explicitly downloaded image batch. Actual item count and normalization are supplied by the caller.
///
/// RU: Нумерованная сетка явно выгруженного батча картинок. Фактическое число элементов и нормализацию задаёт вызывающий.
/// </summary>
public static class TensorBatchPreviewRenderer
{
    public const int HeaderHeight = 24;
    public const int LabelHeight = 20;

    public static CvSize GridSize(CvSize itemSize, int actualCount)
    {
        if(itemSize.Width <= 0 || itemSize.Height <= 0 || actualCount < 0) throw new ArgumentOutOfRangeException(nameof(itemSize));
        var size = new CvSize(checked(itemSize.Width * (actualCount > 1 ? 2 : 1)), checked(HeaderHeight + Math.Max(1, (actualCount + 1) / 2) * (itemSize.Height + LabelHeight)));
        if((long)size.Width * size.Height > 16 * 1024 * 1024) throw new InvalidOperationException("Batch grid exceeds 16 megapixels.");
        return size;
    }

    public static Mat CreateEmpty(CvSize itemSize, int capacity)
    {
        var canvas = new Mat(GridSize(itemSize, 0), MatType.CV_8UC3, Scalar.All(24));
        Cv2.PutText(canvas, $"0 / {capacity}", new CvPoint(6, 17), HersheyFonts.HersheySimplex, 0.5, Scalar.All(240), 1);
        return canvas;
    }

    public static Mat CreatePreview(OrtValue tensor, int actualCount, CvSize itemSize, bool paddleNormalized = false)
    {
        using OrtMemoryInfo memory = tensor.GetTensorMemoryInfo();
        if(!string.Equals(memory.Name, "Cpu", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Batch rendering requires an explicit CPU download.");
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        if(shape.Length != 4 || shape[0] < 1 || actualCount < 0 || actualCount > shape[0])
            throw new ArgumentException("Actual image count must fit the rank-4 batch.");
        if(actualCount == 0) return CreateEmpty(itemSize, checked((int)shape[0]));
        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        var sourceSize = new CvSize(checked((int)shape[nhwc ? 2 : 3]), checked((int)shape[nhwc ? 1 : 2]));
        if(itemSize.Width > sourceSize.Width || itemSize.Height > sourceSize.Height || itemSize.Width <= 0 || itemSize.Height <= 0)
            throw new ArgumentException("Batch preview item size must be positive and cannot upscale the image.");
        int columns = actualCount > 1 ? 2 : 1;
        var canvas = new Mat(GridSize(itemSize, actualCount), MatType.CV_8UC3, Scalar.All(24));
        try
        {
            Cv2.PutText(canvas, $"{actualCount} / {shape[0]}", new CvPoint(6, 17), HersheyFonts.HersheySimplex, 0.5, Scalar.All(240), 1);
            for(int index = 0; index < actualCount; index++)
            {
                int x = index % columns * itemSize.Width;
                int y = HeaderHeight + index / columns * (itemSize.Height + LabelHeight);
                Cv2.PutText(canvas, $"[{index}]", new CvPoint(x + 6, y + 15), HersheyFonts.HersheySimplex, 0.45, Scalar.All(240), 1);
                using Mat item = TensorImagePreviewRenderer.CreateBatchItem(tensor, index, paddleNormalized);
                using var destination = new Mat(canvas, new CvRect(x, y + LabelHeight, itemSize.Width, itemSize.Height));
                if(item.Size() == itemSize) item.CopyTo(destination);
                else Cv2.Resize(item, destination, itemSize);
            }
            return canvas;
        }
        catch { canvas.Dispose(); throw; }
    }
}
