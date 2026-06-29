using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_RealImage;

public sealed class Op_Onnx_RealImageVisualTests
{
    static readonly PerspectiveVisualCase ConveyorDrawingPerspective = new(
        CaseId: "conveyor_drawing",
        ImageFileName: "for_tests_perspective.png",
        OutputSize: new Size(900, 1300),
        SourceQuad:
        [
            new(1050, 195), // 0: far left
            new(1326, 231), // 1: far right
            new(850, 1180), // 2: near right
            new(-120, 700)  // 3: near left; may be negative if the conveyor edge leaves the frame
        ]);

    static readonly PerspectiveVisualCase ConveyorReal0Perspective = new(
        CaseId: "conveyor_real_0",
        ImageFileName: "for_tests_perspective_real_0.png",
        OutputSize: new Size(900, 4500),
        SourceQuad:
        [
            new(918, 0),     // 0: far left
            new(1142, 0),    // 1: far right
            new(1315, 1558), // 2: near right
            new(732, 1558)   // 3: near left
        ]);

    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    public static IEnumerable<object[]> PerspectiveCases =>
        from backend in OnnxExecutionBackendMatrix.EnabledBackends
        from testCase in new[] { ConveyorDrawingPerspective, ConveyorReal0Perspective }
        select new object[] { testCase, backend };

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task CropU8NHWC_SavesRealImageVisualArtifact(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr();
        Rect cropRect = CreateCenteredRect(source, widthRatio: 0.52, heightRatio: 0.55);
        using Mat actual = await RunU8Nhwc(source, executionBackend, (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Crop_U8_NHWC(
            "input",
            "output",
            cropRect,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(cropRect.Width, actual.Width);
        Assert.Equal(cropRect.Height, actual.Height);
        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_Crop_U8_NHWC", "real_for_tests_center_crop");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task ResizeU8NHWC_SavesRealImageVisualArtifact(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr();
        var targetSize = new Size(1000, 667);
        using Mat actual = await RunU8Nhwc(source, executionBackend, static (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize_U8_NHWC(
            "input",
            "output",
            new Size(1000, 667),
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(targetSize.Width, actual.Width);
        Assert.Equal(targetSize.Height, actual.Height);
        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_Resize_U8_NHWC", "real_for_tests_to_1000x667");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task PadResizeU8NHWC_SavesRealImageVisualArtifact(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr();
        var targetSize = new Size(1024, 1024);
        using Mat actual = await RunU8Nhwc(source, executionBackend, static (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_U8_NHWC(
            "input",
            "output",
            new Size(1024, 1024),
            stride: 32,
            mode: PadResizeMode.FixedCanvas,
            padValue: 114,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(targetSize.Width, actual.Width);
        Assert.Equal(targetSize.Height, actual.Height);
        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_PadResize_U8_NHWC", "real_for_tests_to_1024x1024");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task Rotate90U8NHWC_SavesRealImageVisualArtifact(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr();
        using Mat actual = await RunU8Nhwc(source, executionBackend, static (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_U8_NHWC(
            "input",
            "output",
            Rotate90Mode.Clockwise90,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(source.Height, actual.Width);
        Assert.Equal(source.Width, actual.Height);
        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_Rotate90_U8_NHWC", "real_for_tests_clockwise");
    }

    [Theory]
    [MemberData(nameof(PerspectiveCases))]
    public async Task PerspectiveU8NHWC_SavesRealImageVisualArtifact(PerspectiveVisualCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr(testCase.ImageFileName);
        Point2f[] sourceQuad = testCase.CreateSourceQuadCopy();
        Size outputSize = testCase.OutputSize;
        using Mat actual = await RunU8Nhwc(source, executionBackend, (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_U8_NHWC(
            "input",
            "output",
            sourceQuad,
            outputSize,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(outputSize.Width, actual.Width);
        Assert.Equal(outputSize.Height, actual.Height);
        SavePerspectiveArtifacts(source, actual, sourceQuad, executionBackend, "Op_Onnx_Perspective_U8_NHWC", CreatePerspectiveCaseName(testCase, sourceQuad, outputSize));
    }

    [Theory]
    [MemberData(nameof(PerspectiveCases))]
    public async Task PerspectiveFP32NCHW_SavesRealImageVisualArtifact(PerspectiveVisualCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr(testCase.ImageFileName);
        Point2f[] sourceQuad = testCase.CreateSourceQuadCopy();
        Size outputSize = testCase.OutputSize;
        using Mat actual = await RunFP32Nchw(source, executionBackend, (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_FP32_NCHW(
            "input",
            "output",
            sourceQuad,
            outputSize,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(outputSize.Width, actual.Width);
        Assert.Equal(outputSize.Height, actual.Height);
        SavePerspectiveArtifacts(source, actual, sourceQuad, executionBackend, "Op_Onnx_Perspective_FP32_NCHW", CreatePerspectiveCaseName(testCase, sourceQuad, outputSize));
    }

    [Theory]
    [MemberData(nameof(PerspectiveCases))]
    public async Task PerspectiveFP16NCHW_SavesRealImageVisualArtifact(PerspectiveVisualCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr(testCase.ImageFileName);
        Point2f[] sourceQuad = testCase.CreateSourceQuadCopy();
        Size outputSize = testCase.OutputSize;
        using Mat actual = await RunFP16Nchw(source, executionBackend, (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Perspective_FP16_NCHW(
            "input",
            "output",
            sourceQuad,
            outputSize,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(outputSize.Width, actual.Width);
        Assert.Equal(outputSize.Height, actual.Height);
        SavePerspectiveArtifacts(source, actual, sourceQuad, executionBackend, "Op_Onnx_Perspective_FP16_NCHW", CreatePerspectiveCaseName(testCase, sourceQuad, outputSize));
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task UndistortU8NHWC_SavesRealImageVisualArtifact(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat source = RealImageTestSource.LoadBgr();
        var distortion = new RadialTangentialDistortionParameters(
            Fx: source.Width * 0.75f,
            Fy: source.Width * 0.75f,
            Cx: source.Width * 0.5f,
            Cy: source.Height * 0.5f,
            K1: -0.18f,
            K2: 0.04f);
        using Mat actual = await RunU8Nhwc(source, executionBackend, (backend, isFinal) => new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Undistort_U8_NHWC(
            "input",
            "output",
            distortion,
            isFinal: isFinal,
            executionBackend: backend));

        Assert.Equal(source.Width, actual.Width);
        Assert.Equal(source.Height, actual.Height);
        SaveArtifacts(source, actual, executionBackend, "Op_Onnx_Undistort_U8_NHWC", "real_for_tests_barrel");
    }

    static async Task<Mat> RunU8Nhwc(
        Mat source,
        InferenceBackend executionBackend,
        Func<InferenceBackend, bool, global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase> instructionFactory)
    {
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(source);
        await using var context = VmRunContextFactory.Create();
        using global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase instruction = instructionFactory(executionBackend, true);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.UInt8, output.GetTensorTypeAndShape().ElementDataType);
        return OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);
    }

    static async Task<Mat> RunFP32Nchw(
        Mat source,
        InferenceBackend executionBackend,
        Func<InferenceBackend, bool, global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase> instructionFactory)
    {
        using OrtValue input = CreateFP32NchwTensor(source);
        await using var context = VmRunContextFactory.Create();
        using global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase instruction = instructionFactory(executionBackend, true);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float, output.GetTensorTypeAndShape().ElementDataType);
        return CreatePreviewFromFP32Nchw(output);
    }

    static async Task<Mat> RunFP16Nchw(
        Mat source,
        InferenceBackend executionBackend,
        Func<InferenceBackend, bool, global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase> instructionFactory)
    {
        using OrtValue input = CreateFP16NchwTensor(source);
        await using var context = VmRunContextFactory.Create();
        using global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_TensorTransformBase instruction = instructionFactory(executionBackend, true);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        OrtValue output = context.Get<OrtValue>("output");
        Assert.Equal(TensorElementType.Float16, output.GetTensorTypeAndShape().ElementDataType);
        return CreatePreviewFromFP16Nchw(output);
    }

    static OrtValue CreateFP32NchwTensor(Mat source)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float, [1, 3, source.Height, source.Width]);
        Span<float> data = tensor.GetTensorMutableDataAsSpan<float>();

        for(int y = 0; y < source.Height; y++)
        {
            for(int x = 0; x < source.Width; x++)
            {
                Vec3b pixel = source.At<Vec3b>(y, x);
                int offset = y * source.Width + x;
                data[offset] = pixel.Item0 / 255f;
                data[source.Height * source.Width + offset] = pixel.Item1 / 255f;
                data[2 * source.Height * source.Width + offset] = pixel.Item2 / 255f;
            }
        }

        return tensor;
    }

    static OrtValue CreateFP16NchwTensor(Mat source)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, [1, 3, source.Height, source.Width]);
        Span<Half> data = MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorMutableDataAsSpan<Float16>());

        for(int y = 0; y < source.Height; y++)
        {
            for(int x = 0; x < source.Width; x++)
            {
                Vec3b pixel = source.At<Vec3b>(y, x);
                int offset = y * source.Width + x;
                data[offset] = (Half)(pixel.Item0 / 255f);
                data[source.Height * source.Width + offset] = (Half)(pixel.Item1 / 255f);
                data[2 * source.Height * source.Width + offset] = (Half)(pixel.Item2 / 255f);
            }
        }

        return tensor;
    }

    static Mat CreatePreviewFromFP32Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(tensor.GetTensorDataAsSpan<float>(), checked((int)shape[2]), checked((int)shape[3]), static value => ToByte(value));
    }

    static Mat CreatePreviewFromFP16Nchw(OrtValue tensor)
    {
        long[] shape = tensor.GetTensorTypeAndShape().Shape;
        return CreatePreviewFromNchw(MemoryMarshal.Cast<Float16, Half>(tensor.GetTensorDataAsSpan<Float16>()), checked((int)shape[2]), checked((int)shape[3]), static value => ToByte((float)value));
    }

    static Mat CreatePreviewFromNchw<T>(ReadOnlySpan<T> data, int height, int width, Func<T, byte> convert)
    {
        var image = new Mat(height, width, MatType.CV_8UC3);
        int planeSize = checked(height * width);

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                int offset = y * width + x;
                image.Set(y, x, new Vec3b(
                    convert(data[offset]),
                    convert(data[planeSize + offset]),
                    convert(data[2 * planeSize + offset])));
            }
        }

        return image;
    }

    static byte ToByte(float value) =>
        (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);

    static string CreatePerspectiveCaseName(PerspectiveVisualCase testCase, IReadOnlyList<Point2f> sourceQuad, Size outputSize)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"real_perspective_{testCase.CaseId}_out_{outputSize.Width}x{outputSize.Height}_q0_{sourceQuad[0].X:F0}_{sourceQuad[0].Y:F0}_q1_{sourceQuad[1].X:F0}_{sourceQuad[1].Y:F0}_q2_{sourceQuad[2].X:F0}_{sourceQuad[2].Y:F0}_q3_{sourceQuad[3].X:F0}_{sourceQuad[3].Y:F0}");
    }

    static Rect CreateCenteredRect(Mat source, double widthRatio, double heightRatio)
    {
        int width = Math.Max(1, (int)Math.Round(source.Width * widthRatio));
        int height = Math.Max(1, (int)Math.Round(source.Height * heightRatio));
        return new Rect((source.Width - width) / 2, (source.Height - height) / 2, width, height);
    }

    static void SaveArtifacts(Mat source, Mat actual, InferenceBackend executionBackend, string operationName, string caseName)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;
        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} source: {caseName}",
            source,
            environment.CreateOperationArtifactPath("Onnx", operationName, executionBackend.ToString(), caseName, "source"));
        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} actual: {caseName}",
            actual,
            environment.CreateOperationArtifactPath("Onnx", operationName, executionBackend.ToString(), caseName, "actual"));
    }

    static void SavePerspectiveArtifacts(
        Mat source,
        Mat actual,
        IReadOnlyList<Point2f> sourceQuad,
        InferenceBackend executionBackend,
        string operationName,
        string caseName)
    {
        SaveArtifacts(source, actual, executionBackend, operationName, caseName);

        using Mat sourceWithQuad = CreateQuadOverlayCanvas(source, sourceQuad);

        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;
        VisualTestOutput.SaveOrShow(
            $"{operationName} {executionBackend} source quad: {caseName}",
            sourceWithQuad,
            environment.CreateOperationArtifactPath("Onnx", operationName, executionBackend.ToString(), caseName, "source_quad"));
    }

    static Mat CreateQuadOverlayCanvas(Mat source, IReadOnlyList<Point2f> quad)
    {
        Rect bounds = CreateQuadBounds(source, quad);
        var canvas = new Mat(bounds.Height, bounds.Width, source.Type(), new Scalar(20, 20, 20));
        var imageOffset = new Point(-bounds.X, -bounds.Y);
        using var targetRoi = new Mat(canvas, new Rect(imageOffset.X, imageOffset.Y, source.Width, source.Height));
        source.CopyTo(targetRoi);

        Cv2.Rectangle(
            canvas,
            new Rect(imageOffset.X, imageOffset.Y, source.Width, source.Height),
            new Scalar(255, 255, 255),
            thickness: 2,
            lineType: LineTypes.AntiAlias);

        DrawQuad(canvas, quad, imageOffset);
        return canvas;
    }

    static Rect CreateQuadBounds(Mat source, IReadOnlyList<Point2f> quad)
    {
        int minX = 0;
        int minY = 0;
        int maxX = source.Width;
        int maxY = source.Height;

        foreach(Point2f point in quad)
        {
            minX = Math.Min(minX, (int)MathF.Floor(point.X));
            minY = Math.Min(minY, (int)MathF.Floor(point.Y));
            maxX = Math.Max(maxX, (int)MathF.Ceiling(point.X));
            maxY = Math.Max(maxY, (int)MathF.Ceiling(point.Y));
        }

        const int margin = 100;
        minX -= margin;
        minY -= margin;
        maxX += margin;
        maxY += margin;

        return new Rect(minX, minY, checked(maxX - minX), checked(maxY - minY));
    }

    static void DrawQuad(Mat image, IReadOnlyList<Point2f> quad, Point offset)
    {
        if(quad.Count < 4)
            return;

        Point[] points =
        [
            new((int)MathF.Round(quad[0].X) + offset.X, (int)MathF.Round(quad[0].Y) + offset.Y),
            new((int)MathF.Round(quad[1].X) + offset.X, (int)MathF.Round(quad[1].Y) + offset.Y),
            new((int)MathF.Round(quad[2].X) + offset.X, (int)MathF.Round(quad[2].Y) + offset.Y),
            new((int)MathF.Round(quad[3].X) + offset.X, (int)MathF.Round(quad[3].Y) + offset.Y)
        ];

        Cv2.Polylines(image, [points], isClosed: true, Scalar.Lime, thickness: 5, lineType: LineTypes.AntiAlias);
        for(int index = 0; index < points.Length; index++)
        {
            Cv2.Circle(image, points[index], 12, Scalar.Red, thickness: -1, lineType: LineTypes.AntiAlias);
            Cv2.PutText(
                image,
                string.Create(CultureInfo.InvariantCulture, $"{index}: {quad[index].X:F0},{quad[index].Y:F0}"),
                points[index] + new Point(14, -14),
                HersheyFonts.HersheySimplex,
                0.9,
                Scalar.Yellow,
                thickness: 3,
                lineType: LineTypes.AntiAlias);
        }
    }

    public sealed record PerspectiveVisualCase(
        string CaseId,
        string ImageFileName,
        Size OutputSize,
        Point2f[] SourceQuad)
    {
        public Point2f[] CreateSourceQuadCopy() => [.. SourceQuad];

        public override string ToString() => CaseId;
    }
}
