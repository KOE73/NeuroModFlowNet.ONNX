using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Rotate90;

internal static class OpOnnxRotate90Pattern
{
    static readonly Vec3b Background = new(19, 23, 31);
    static readonly Vec3b TopLeftMarker = new(9, 73, 201);
    static readonly Vec3b TopRightMarker = new(31, 211, 47);
    static readonly Vec3b BottomLeftMarker = new(223, 67, 11);
    static readonly Vec3b BottomRightMarker = new(241, 241, 241);

    public static IEnumerable<object[]> DeterministicCases()
    {
        yield return [CreateCornerMarkerCase(960, 640, Rotate90Mode.Clockwise90)];
        yield return [CreateCornerMarkerCase(960, 640, Rotate90Mode.Rotate180)];
        yield return [CreateCornerMarkerCase(960, 640, Rotate90Mode.CounterClockwise90)];
    }

    public static OpOnnxRotate90PatternCase CreateCornerMarkerCase(int width, int height, Rotate90Mode mode)
    {
        Mat source = CreateCoordinateBackground(width, height);
        source.Set(0, 0, TopLeftMarker);
        source.Set(0, width - 1, TopRightMarker);
        source.Set(height - 1, 0, BottomLeftMarker);
        source.Set(height - 1, width - 1, BottomRightMarker);

        Size expectedSize = mode == Rotate90Mode.Rotate180
            ? new Size(width, height)
            : new Size(height, width);

        return new OpOnnxRotate90PatternCase
        {
            Name = $"corner_marker_{width}x{height}_{mode}",
            Source = source,
            Mode = mode,
            ExpectedSize = expectedSize,
            ExpectedPixels = CreateExpectedPixels(expectedSize, mode)
        };
    }

    public static void AssertResult(Mat actual, OpOnnxRotate90PatternCase testCase)
    {
        VisualAssert.MatSize(actual, testCase.ExpectedSize.Width, testCase.ExpectedSize.Height);

        foreach(PixelExpectation expectation in testCase.ExpectedPixels)
            VisualAssert.PixelBgrNear(actual, expectation, testCase.Tolerance);
    }

    public static void SaveArtifacts(Mat actual, OpOnnxRotate90PatternCase testCase, InferenceBackend executionBackend)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Rotate90 {executionBackend} source: {testCase.Name}",
            testCase.Source,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Rotate90_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "source"));

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Rotate90 {executionBackend} actual: {testCase.Name}",
            actual,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Rotate90_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "actual"));
    }

    static PixelExpectation[] CreateExpectedPixels(Size expectedSize, Rotate90Mode mode)
    {
        int right = expectedSize.Width - 1;
        int bottom = expectedSize.Height - 1;

        return mode switch
        {
            Rotate90Mode.Clockwise90 =>
            [
                new PixelExpectation(0, 0, BottomLeftMarker, "top-left rotated pixel"),
                new PixelExpectation(right, 0, TopLeftMarker, "top-right rotated pixel"),
                new PixelExpectation(0, bottom, BottomRightMarker, "bottom-left rotated pixel"),
                new PixelExpectation(right, bottom, TopRightMarker, "bottom-right rotated pixel")
            ],
            Rotate90Mode.Rotate180 =>
            [
                new PixelExpectation(0, 0, BottomRightMarker, "top-left rotated pixel"),
                new PixelExpectation(right, 0, BottomLeftMarker, "top-right rotated pixel"),
                new PixelExpectation(0, bottom, TopRightMarker, "bottom-left rotated pixel"),
                new PixelExpectation(right, bottom, TopLeftMarker, "bottom-right rotated pixel")
            ],
            Rotate90Mode.CounterClockwise90 =>
            [
                new PixelExpectation(0, 0, TopRightMarker, "top-left rotated pixel"),
                new PixelExpectation(right, 0, BottomRightMarker, "top-right rotated pixel"),
                new PixelExpectation(0, bottom, TopLeftMarker, "bottom-left rotated pixel"),
                new PixelExpectation(right, bottom, BottomLeftMarker, "bottom-right rotated pixel")
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported rotate mode.")
        };
    }

    static Mat CreateCoordinateBackground(int width, int height)
    {
        var source = new Mat(height, width, MatType.CV_8UC3, Background.ToScalar());

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                if(x % 16 == 0 || y % 16 == 0)
                    source.Set(y, x, new Vec3b((byte)x, (byte)y, (byte)((x + y) & 0xFF)));
            }
        }

        return source;
    }

    static Scalar ToScalar(this Vec3b color) => new(color.Item0, color.Item1, color.Item2);
}
