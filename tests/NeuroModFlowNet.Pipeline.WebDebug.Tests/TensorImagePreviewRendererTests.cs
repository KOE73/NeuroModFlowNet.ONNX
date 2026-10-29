using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class TensorImagePreviewRendererTests
{
    [Fact]
    public void BgrNhwcPreviewPreservesChannelsAndDimensions()
    {
        const int width = 600, height = 500;
        var pixels = new byte[width * height * 3];
        for(int index = 0; index < width * height; index++)
        {
            pixels[index * 3] = 17;
            pixels[index * 3 + 1] = 90;
            pixels[index * 3 + 2] = 201;
        }
        using OrtValue tensor = OrtValue.CreateTensorValueFromMemory(pixels,new long[] {1,height,width,3});
        byte[] encoded = TensorImagePreviewRenderer.EncodePng(tensor);
        using Mat actual = Cv2.ImDecode(encoded,ImreadModes.Color);
        Assert.Equal(width,actual.Width);
        Assert.Equal(height,actual.Height);
        Assert.Equal(new Vec3b(17,90,201),actual.At<Vec3b>(250,300));
    }

    [Fact]
    public void RgbFp32NchwPreviewUsesDeclaredRangeWithoutAutoContrast()
    {
        const int width = 600, height = 500;
        var pixels = new float[3 * width * height];
        Array.Fill(pixels, 1f, 0, width * height);
        Array.Fill(pixels, 0.5f, width * height, width * height);
        using OrtValue tensor = OrtValue.CreateTensorValueFromMemory(pixels,new long[] {1,3,height,width});
        using Mat actual = Cv2.ImDecode(TensorImagePreviewRenderer.EncodePng(tensor),ImreadModes.Color);
        Assert.Equal(new Vec3b(0,128,255),actual.At<Vec3b>(250,300));
    }
}
