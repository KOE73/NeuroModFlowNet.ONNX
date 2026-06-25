using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class OrtTestTensorFactory
{
    public static OrtValue CreateBgrU8NhwcTensor(Mat source)
    {
        if(source.Type() != MatType.CV_8UC3)
            throw new ArgumentException("Source image must be CV_8UC3.", nameof(source));

        Mat continuousSource = source.IsContinuous()
            ? source
            : source.Clone();

        long[] shape = [1, continuousSource.Height, continuousSource.Width, 3];
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.UInt8, shape);
        Span<byte> tensorData = tensor.GetTensorMutableDataAsSpan<byte>();

        unsafe
        {
            var sourceData = new ReadOnlySpan<byte>(
                (void*)continuousSource.DataPointer,
                checked(continuousSource.Height * continuousSource.Width * 3));
            sourceData.CopyTo(tensorData);
        }

        if(!ReferenceEquals(continuousSource, source))
            continuousSource.Dispose();

        return tensor;
    }

    public static Mat CreateBgrMatFromNhwcTensor(OrtValue tensor)
    {
        var info = tensor.GetTensorTypeAndShape();

        if(info.ElementDataType != TensorElementType.UInt8)
            throw new ArgumentException($"Tensor must be UInt8, actual: {info.ElementDataType}.", nameof(tensor));

        if(info.Shape.Length != 4 || info.Shape[0] != 1 || info.Shape[3] != 3)
            throw new ArgumentException($"Tensor must be NHWC with one batch and three channels, actual: [{string.Join(", ", info.Shape)}].", nameof(tensor));

        int height = checked((int)info.Shape[1]);
        int width = checked((int)info.Shape[2]);
        var image = new Mat(height, width, MatType.CV_8UC3);
        ReadOnlySpan<byte> tensorData = tensor.GetTensorDataAsSpan<byte>();

        unsafe
        {
            var imageData = new Span<byte>(
                (void*)image.DataPointer,
                checked(height * width * 3));
            tensorData.CopyTo(imageData);
        }

        return image;
    }
}
