using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Resize;

internal static class OpOnnxResizePattern
{
    static readonly Vec3b TopLeftColor = new(12, 34, 56);
    static readonly Vec3b TopRightColor = new(190, 48, 24);
    static readonly Vec3b BottomLeftColor = new(24, 180, 70);
    static readonly Vec3b BottomRightColor = new(220, 220, 36);

    public static IEnumerable<object[]> DeterministicCases()
    {
        yield return [CreateQuadrantCase(512, 384, new Size(1024, 768))];
        yield return [CreateQuadrantCase(1000, 750, new Size(500, 375))];
    }

    static OpOnnxResizePatternCase CreateQuadrantCase(int width, int height, Size targetSize)
    {
        var source = new Mat(height, width, MatType.CV_8UC3);
        int halfWidth = width / 2;
        int halfHeight = height / 2;

        source[new Rect(0, 0, halfWidth, halfHeight)].SetTo(ToScalar(TopLeftColor));
        source[new Rect(halfWidth, 0, width - halfWidth, halfHeight)].SetTo(ToScalar(TopRightColor));
        source[new Rect(0, halfHeight, halfWidth, height - halfHeight)].SetTo(ToScalar(BottomLeftColor));
        source[new Rect(halfWidth, halfHeight, width - halfWidth, height - halfHeight)].SetTo(ToScalar(BottomRightColor));

        return new OpOnnxResizePatternCase
        {
            Name = $"quadrants_{width}x{height}_to_{targetSize.Width}x{targetSize.Height}",
            Source = source,
            TargetSize = targetSize,
            ExpectedPixels =
            [
                new PixelExpectation(targetSize.Width / 4, targetSize.Height / 4, TopLeftColor, "top-left quadrant"),
                new PixelExpectation((targetSize.Width * 3) / 4, targetSize.Height / 4, TopRightColor, "top-right quadrant"),
                new PixelExpectation(targetSize.Width / 4, (targetSize.Height * 3) / 4, BottomLeftColor, "bottom-left quadrant"),
                new PixelExpectation((targetSize.Width * 3) / 4, (targetSize.Height * 3) / 4, BottomRightColor, "bottom-right quadrant")
            ]
        };
    }

    public static void AssertResult(Mat actual, OpOnnxResizePatternCase testCase)
    {
        VisualAssert.MatSize(actual, testCase.TargetSize.Width, testCase.TargetSize.Height);

        foreach(PixelExpectation expectation in testCase.ExpectedPixels)
            VisualAssert.PixelBgrNear(actual, expectation, testCase.Tolerance);
    }

    public static void SaveArtifacts(Mat actual, OpOnnxResizePatternCase testCase, InferenceBackend executionBackend)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Resize {executionBackend} source: {testCase.Name}",
            testCase.Source,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Resize",
                executionBackend.ToString(),
                testCase.Name,
                "source"));

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Resize {executionBackend} actual: {testCase.Name}",
            actual,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Resize",
                executionBackend.ToString(),
                testCase.Name,
                "actual"));
    }

    static Scalar ToScalar(Vec3b color) => new(color.Item0, color.Item1, color.Item2);
}
