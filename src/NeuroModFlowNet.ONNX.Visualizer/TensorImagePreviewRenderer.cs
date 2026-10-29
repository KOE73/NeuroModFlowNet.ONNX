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
        using OrtMemoryInfo memory = tensor.GetTensorMemoryInfo();
        if(!string.Equals(memory.Name, "Cpu", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Preview rendering requires an explicit download to CPU memory.");
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        if(shape.Length != 4 || shape[0] != 1)
            throw new NotSupportedException("Preview requires one rank-4 image; batch tensors are metadata-only.");

        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        if((nhwc ? shape[3] : shape[1]) != 3)
            throw new NotSupportedException("Preview requires BGR U8 NHWC or RGB FP16/FP32 NCHW with three channels.");
        int height = checked((int)shape[nhwc ? 1 : 2]);
        int width = checked((int)shape[nhwc ? 2 : 3]);
        using var image = new Mat(height, width, MatType.CV_8UC3);
        unsafe
        {
            var destination = new Span<byte>((void*)image.Data, checked(width * height * 3));
            if(nhwc)
                tensor.GetTensorDataAsSpan<byte>().CopyTo(destination);
            else if(information.ElementDataType == TensorElementType.Float)
            {
                ReadOnlySpan<float> source = tensor.GetTensorDataAsSpan<float>();
                for(int index = 0; index < width * height; index++)
                    for(int channel = 0; channel < 3; channel++)
                        destination[index * 3 + (2 - channel)] = ToByte(source[channel * width * height + index], paddleNormalized);
            }
            else if(information.ElementDataType == TensorElementType.Float16)
            {
                ReadOnlySpan<Float16> source = tensor.GetTensorDataAsSpan<Float16>();
                for(int index = 0; index < width * height; index++)
                    for(int channel = 0; channel < 3; channel++)
                        destination[index * 3 + (2 - channel)] = ToByte((float)source[channel * width * height + index], paddleNormalized);
            }
            else
                throw new NotSupportedException($"Unsupported image element type: {information.ElementDataType}.");
        }
        Cv2.ImEncode(".png", image, out byte[] encoded);
        return encoded;
    }

    public static byte[] EncodePng(Mat image, int maxEdge = 1000)
    {
        if(maxEdge is < 100 or > 2000) throw new ArgumentOutOfRangeException(nameof(maxEdge));
        double scale = Math.Min(1, maxEdge / (double)Math.Max(image.Width, image.Height));
        if(scale >= 1) { Cv2.ImEncode(".png", image, out byte[] original); return original; }
        using var resized = new Mat();
        Cv2.Resize(image, resized, new CvSize(Math.Max(1, (int)(image.Width * scale)), Math.Max(1, (int)(image.Height * scale))));
        Cv2.ImEncode(".png", resized, out byte[] encoded);
        return encoded;
    }

    static byte ToByte(float value, bool paddleNormalized) =>
        (byte)Math.Clamp((int)MathF.Round((paddleNormalized ? (value + 1) * 0.5f : value) * 255), 0, 255);
}
