using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Encodes an explicitly downloaded single RGB NCHW or BGR NHWC image for debugger inspection.
///
/// RU: Кодирует явно выгруженное одиночное изображение RGB NCHW или BGR NHWC для отладчика.
/// </summary>
/// <remarks>
/// EN: Never reads device pointers. Floating point inputs use the caller's declared range; no automatic contrast adjustment.
///
/// RU: Не читает указатели устройства. Диапазон floating point задаёт вызывающий; автоматической коррекции контраста нет.
/// </remarks>
public static class TensorImagePreviewRenderer
{
    public static byte[] EncodePng(OrtValue tensor, bool paddleNormalized = false)
    {
        using Mat image = CreatePreview(tensor, paddleNormalized);
        Cv2.ImEncode(".png", image, out byte[] encoded);
        return encoded;
    }

    /// <summary>
    /// EN: Copies an already downloaded tensor into an owned BGR image, independent of the VM lifetime.
    ///
    /// RU: Копирует уже выгруженный тензор во владеющее BGR-изображение, независимое от времени жизни VM.
    /// </summary>
    /// <remarks>
    /// EN: The caller must dispose the returned image. Encoding may then run on another thread.
    ///
    /// RU: Вызывающий освобождает возвращённое изображение. После копирования кодирование допустимо в другом потоке.
    /// </remarks>
    public static Mat CreatePreview(OrtValue tensor, bool paddleNormalized = false)
        => CreateImage(tensor, 0, true, paddleNormalized);

    /// <summary>
    /// EN: Copies one explicitly indexed item of an already downloaded image batch into an owned BGR image.
    ///
    /// RU: Копирует элемент с явно заданным индексом из уже выгруженного батча во владеющее BGR-изображение.
    /// </summary>
    public static Mat CreateBatchItem(OrtValue tensor, int batchIndex, bool paddleNormalized = false)
        => CreateImage(tensor, batchIndex, false, paddleNormalized);

    static Mat CreateImage(OrtValue tensor, int batchIndex, bool requireSingle, bool paddleNormalized)
    {
        using OrtMemoryInfo memory = tensor.GetTensorMemoryInfo();
        if(!string.Equals(memory.Name, "Cpu", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Preview rendering requires an explicit download to CPU memory.");
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        if(shape.Length != 4 || shape[0] <= 0 || requireSingle && shape[0] != 1)
            throw new NotSupportedException("Preview requires one rank-4 image; batch tensors are metadata-only.");
        if(batchIndex < 0 || batchIndex >= shape[0]) throw new ArgumentOutOfRangeException(nameof(batchIndex));

        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        if((nhwc ? shape[3] : shape[1]) != 3)
            throw new NotSupportedException("Preview requires BGR U8 NHWC or RGB FP16/FP32 NCHW with three channels.");
        int height = checked((int)shape[nhwc ? 1 : 2]);
        int width = checked((int)shape[nhwc ? 2 : 3]);
        int itemLength = checked(width * height * 3);
        int offset = checked(batchIndex * itemLength);
        var image = new Mat(height, width, MatType.CV_8UC3);
        try
        {
            unsafe
            {
                var destination = new Span<byte>((void*)image.Data, checked(width * height * 3));
                if(nhwc)
                    tensor.GetTensorDataAsSpan<byte>().Slice(offset, itemLength).CopyTo(destination);
                else if(information.ElementDataType == TensorElementType.Float)
                {
                    ReadOnlySpan<float> source = tensor.GetTensorDataAsSpan<float>().Slice(offset, itemLength);
                    for(int index = 0; index < width * height; index++)
                        for(int channel = 0; channel < 3; channel++)
                            destination[index * 3 + (2 - channel)] = ToByte(source[channel * width * height + index], paddleNormalized);
                }
                else if(information.ElementDataType == TensorElementType.Float16)
                {
                    ReadOnlySpan<Float16> source = tensor.GetTensorDataAsSpan<Float16>().Slice(offset, itemLength);
                    for(int index = 0; index < width * height; index++)
                        for(int channel = 0; channel < 3; channel++)
                            destination[index * 3 + (2 - channel)] = ToByte((float)source[channel * width * height + index], paddleNormalized);
                }
                else
                    throw new NotSupportedException($"Unsupported image element type: {information.ElementDataType}.");
            }
            return image;
        }
        catch { image.Dispose(); throw; }
    }

    public static byte[] EncodePng(Mat image, int maxEdge = 1000)
    {
        if(maxEdge is < 100 or > 2000) throw new ArgumentOutOfRangeException(nameof(maxEdge));
        if(Math.Max(image.Width, image.Height) <= maxEdge)
        {
            Cv2.ImEncode(".png", image, out byte[] original);
            return original;
        }
        using Mat preview = CreatePreview(image, maxEdge);
        Cv2.ImEncode(".png", preview, out byte[] encoded);
        return encoded;
    }

    /// <summary>
    /// EN: Returns an owned, size-limited image for encoding after the source is released.
    ///
    /// RU: Возвращает владеющее изображение ограниченного размера для кодирования после освобождения источника.
    /// </summary>
    public static Mat CreatePreview(Mat image, int maxEdge = 1000)
    {
        if(maxEdge is < 100 or > 2000) throw new ArgumentOutOfRangeException(nameof(maxEdge));
        double scale = Math.Min(1, maxEdge / (double)Math.Max(image.Width, image.Height));
        if(scale >= 1) return image.Clone();
        var resized = new Mat();
        try
        {
            Cv2.Resize(image, resized, new CvSize(Math.Max(1, (int)(image.Width * scale)), Math.Max(1, (int)(image.Height * scale))));
            return resized;
        }
        catch { resized.Dispose(); throw; }
    }

    static byte ToByte(float value, bool paddleNormalized) =>
        (byte)Math.Clamp((int)MathF.Round((paddleNormalized ? (value + 1) * 0.5f : value) * 255), 0, 255);
}
