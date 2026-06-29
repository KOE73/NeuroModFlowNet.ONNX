using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_PadResize;

internal static class OpOnnxPadResizePattern
{
    static readonly Vec3b ContentColor = new(7, 149, 231);
    static readonly Vec3b PadColor = new(114, 114, 114);

    public static IEnumerable<object[]> DeterministicCases()
    {
        yield return [CreatePanoramicNoPaddingCase()];
        yield return [CreateFixedCanvasPaddingCase()];
        yield return [CreateAutoStrideCase()];
    }

    static OpOnnxPadResizePatternCase CreatePanoramicNoPaddingCase()
    {
        Mat source = CreateSolidImage(2048, 256, ContentColor);
        return new OpOnnxPadResizePatternCase
        {
            Name = "panoramic_2048x256_fixed_no_padding",
            Source = source,
            TargetSize = new Size(2048, 256),
            Mode = PadResizeMode.FixedCanvas,
            ExpectedSize = new Size(2048, 256),
            ExpectedPixels =
            [
                new PixelExpectation(0, 0, ContentColor, "top-left content"),
                new PixelExpectation(2047, 255, ContentColor, "bottom-right content"),
                new PixelExpectation(1024, 128, ContentColor, "center content")
            ]
        };
    }

    static OpOnnxPadResizePatternCase CreateFixedCanvasPaddingCase()
    {
        Mat source = CreateSolidImage(1000, 500, ContentColor);
        return new OpOnnxPadResizePatternCase
        {
            Name = "wide_1000x500_to_640x640_fixed_padding",
            Source = source,
            TargetSize = new Size(640, 640),
            Mode = PadResizeMode.FixedCanvas,
            ExpectedSize = new Size(640, 640),
            ExpectedPixels =
            [
                new PixelExpectation(10, 10, PadColor, "top padding"),
                new PixelExpectation(10, 629, PadColor, "bottom padding"),
                new PixelExpectation(10, 160, ContentColor, "first content row"),
                new PixelExpectation(320, 320, ContentColor, "center content")
            ]
        };
    }

    static OpOnnxPadResizePatternCase CreateAutoStrideCase()
    {
        Mat source = CreateSolidImage(1000, 400, ContentColor);
        return new OpOnnxPadResizePatternCase
        {
            Name = "wide_1000x400_to_641x641_auto_stride",
            Source = source,
            TargetSize = new Size(641, 641),
            Mode = PadResizeMode.AutoStrideCanvas,
            ExpectedSize = new Size(672, 256),
            ExpectedPixels =
            [
                new PixelExpectation(0, 10, PadColor, "left stride padding"),
                new PixelExpectation(671, 10, PadColor, "right stride padding"),
                new PixelExpectation(15, 10, ContentColor, "first content column"),
                new PixelExpectation(335, 128, ContentColor, "center content")
            ]
        };
    }

    public static void AssertResult(Mat actual, OpOnnxPadResizePatternCase testCase)
    {
        VisualAssert.MatSize(actual, testCase.ExpectedSize.Width, testCase.ExpectedSize.Height);

        foreach(PixelExpectation expectation in testCase.ExpectedPixels)
            VisualAssert.PixelBgrNear(actual, expectation, testCase.Tolerance);
    }

    public static void SaveArtifacts(Mat actual, OpOnnxPadResizePatternCase testCase, InferenceBackend executionBackend)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_PadResize {executionBackend} source: {testCase.Name}",
            testCase.Source,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_PadResize_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "source"));

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_PadResize {executionBackend} actual: {testCase.Name}",
            actual,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_PadResize_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "actual"));
    }

    static Mat CreateSolidImage(int width, int height, Vec3b color) =>
        new(height, width, MatType.CV_8UC3, color.ToScalar());

    static Scalar ToScalar(this Vec3b color) => new(color.Item0, color.Item1, color.Item2);
}
