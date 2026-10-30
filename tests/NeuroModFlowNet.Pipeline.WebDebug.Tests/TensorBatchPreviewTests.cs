using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class TensorBatchPreviewTests
{
    [Fact]
    public async Task PaddleGridUsesActualCountAndRecoversRgbColorsFromMinusOneToOne()
    {
        const int width = 640, height = 48, capacity = 16, plane = width * height;
        float[] values = new float[capacity * plane * 3];
        Array.Fill(values, -1);
        Array.Fill(values, 1, 0, plane); // First ROI is red in RGB NCHW.
        Array.Fill(values, 1, 3 * plane + plane, plane); // Second ROI is green.
        Array.Fill(values, 1, 6 * plane + 2 * plane, plane); // Third ROI is blue.
        CapturedRegister captured;
        using(var context = Context())
        using(OrtValue batch = OrtValue.CreateTensorValueFromMemory(values, new long[] { capacity, 3, height, width }))
        {
            context.Set("ocr.recInput", batch); context.Set("ocr.roiCount", 3);
            captured = await Inspector().CaptureAsync(context, new("ocr.recInput", 0, "png", 0), TestContext.Current.CancellationToken);
        }
        Array.Clear(values);
        using(captured)
        using(Mat image = Cv2.ImDecode(captured.Encode().Data, ImreadModes.Color))
        {
            Assert.Equal(new Size(1280, 160), image.Size());
            Assert.Equal(new Vec3b(0, 0, 255), image.At<Vec3b>(60, 320));
            Assert.Equal(new Vec3b(0, 255, 0), image.At<Vec3b>(60, 960));
            Assert.Equal(new Vec3b(255, 0, 0), image.At<Vec3b>(128, 320));
            string root = Path.Combine(Path.GetTempPath(), "NeuroModFlowNet.WebDebug.Tests");
            Directory.CreateDirectory(root);
            Cv2.ImWrite(Path.Combine(root, "paddle-batch-grid.png"), image);
        }
    }

    [Fact]
    public async Task MissingOrInvalidActualCountIsRejectedInsteadOfShowingPadding()
    {
        using var context = Context();
        using OrtValue batch = OrtValue.CreateTensorValueFromMemory(new float[2 * 3 * 48 * 640], new long[] { 2, 3, 48, 640 });
        context.Set("ocr.recInput", batch);
        var inspector = Inspector();
        var request = new DebugInspectionRequest("ocr.recInput", 0, "png", 640);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await inspector.CaptureAsync(context, request, TestContext.Current.CancellationToken));
        context.Set("ocr.roiCount", 3);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await inspector.CaptureAsync(context, request, TestContext.Current.CancellationToken));
        context.Set("ocr.roiCount", 0);
        using CapturedRegister empty = await inspector.CaptureAsync(context, request, TestContext.Current.CancellationToken);
        using Mat image = Cv2.ImDecode(empty.Encode().Data, ImreadModes.Color);
        Assert.Equal(new Size(640, 92), image.Size());
    }

    [Fact]
    public async Task BgrCropBatchKeepsNumberingAndDoesNotApplyPaddleNormalization()
    {
        const int width = 640, height = 480, length = width * height * 3;
        byte[] values = new byte[length * 4];
        Array.Fill(values, (byte)90, length, length);
        using var context = Context();
        using OrtValue batch = OrtValue.CreateTensorValueFromMemory(values, new long[] { 4, height, width, 3 });
        context.Set("ocr.crop.batch", batch); context.Set("ocr.crop.count", 2);
        using CapturedRegister captured = await Inspector().CaptureAsync(context, new("ocr.crop.batch", 0, "png", 320), TestContext.Current.CancellationToken);
        using Mat image = Cv2.ImDecode(captured.Encode().Data, ImreadModes.Color);
        Assert.Equal(new Size(640, 284), image.Size());
        Assert.Equal(new Vec3b(0, 0, 0), image.At<Vec3b>(100, 100));
        Assert.Equal(new Vec3b(90, 90, 90), image.At<Vec3b>(100, 420));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorImagePreviewRenderer.CreateBatchItem(batch, 4));
        Assert.Throws<InvalidOperationException>(() => TensorBatchPreviewRenderer.GridSize(new Size(4000, 4000), 16));
    }

    [Fact]
    public async Task Fp16BatchReadsTheCorrectItemAndChannelOffsets()
    {
        const int plane = 640 * 48;
        var values = new Float16[2 * 3 * plane];
        Array.Fill(values, (Float16)1f, 3 * plane, plane);
        Array.Fill(values, (Float16)0.5f, 5 * plane, plane);
        using var context = Context();
        using OrtValue batch = OrtValue.CreateTensorValueFromMemory(values, new long[] { 2, 3, 48, 640 });
        context.Set("images", batch);
        using CapturedRegister captured = await Inspector().CaptureAsync(context, new("images", 0, "png", 320), TestContext.Current.CancellationToken);
        using Mat image = Cv2.ImDecode(captured.Encode().Data, ImreadModes.Color);
        Assert.Equal(new Size(640, 68), image.Size());
        Assert.Equal(new Vec3b(0, 0, 0), image.At<Vec3b>(55, 100));
        Assert.Equal(new Vec3b(128, 0, 255), image.At<Vec3b>(55, 420));
    }

    static PausedRegisterInspector Inspector() => new(InferenceBackend.TensorRt, Nv12ColorMatrix.Bt601Full);
    static VmRunContext Context() => new(new("camera", 1, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());
}
