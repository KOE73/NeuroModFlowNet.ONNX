using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class PreviewImageSizeTests
{
    [Theory]
    [InlineData(640, 640, 360)]
    [InlineData(2000, 2000, 1125)]
    [InlineData(3840, 3840, 2160)]
    [InlineData(0, 3840, 2160)]
    public void PreviewLimitPreservesAspectAndOriginalKeepsFourK(int maxEdge, int width, int height) =>
        Assert.Equal(new Size(width, height), PreviewImageSize.Resolve(3840, 2160, maxEdge));

    [Fact]
    public void OriginalCanRepresentTheTallTransformedFrameWithoutUpscaling()
    {
        Assert.Equal(new Size(1120, 5408), PreviewImageSize.Resolve(1120, 5408, 0));
        Assert.Equal(new Size(640, 360), PreviewImageSize.Resolve(640, 360, 3840));
        Assert.Throws<InvalidOperationException>(() => PreviewImageSize.Resolve(10000, 10000, 0));
    }

    [Fact]
    public async Task OriginalCpuTensorIsOwnedAndKeepsFourKPixels()
    {
        const int width = 3840, height = 2160;
        byte[] pixels = new byte[width * height * 3];
        pixels[0] = 17; pixels[1] = 90; pixels[2] = 201;
        CapturedRegister captured;
        using(var context = new VmRunContext(new("camera", 1, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry()))
        using(OrtValue tensor = OrtValue.CreateTensorValueFromMemory(pixels, new long[] { 1, height, width, 3 }))
        {
            context.Set("image", tensor);
            var inspector = new PausedRegisterInspector(InferenceBackend.TensorRt, Nv12ColorMatrix.Bt601Full);
            captured = await inspector.CaptureAsync(context, new("image", 0, "png", 0), TestContext.Current.CancellationToken);
        }
        Array.Clear(pixels);
        using(captured)
        using(Mat image = Cv2.ImDecode(captured.Encode().Data, ImreadModes.Color))
        {
            Assert.Equal(new Size(width, height), image.Size());
            Assert.Equal(new Vec3b(17, 90, 201), image.At<Vec3b>(0, 0));
        }
    }
}
