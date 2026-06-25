using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class VisualAssert
{
    public static void MatSize(Mat actual, int width, int height)
    {
        Assert.Equal(width, actual.Width);
        Assert.Equal(height, actual.Height);
    }

    public static void PixelBgrNear(Mat actual, PixelExpectation expectation, int tolerance = 0)
    {
        Vec3b actualBgr = actual.At<Vec3b>(expectation.Y, expectation.X);
        BgrNear(actualBgr, expectation.ExpectedBgr, tolerance, expectation.Name);
    }

    public static void BgrNear(Vec3b actual, Vec3b expected, int tolerance, string name)
    {
        Assert.True(
            Math.Abs(actual.Item0 - expected.Item0) <= tolerance &&
            Math.Abs(actual.Item1 - expected.Item1) <= tolerance &&
            Math.Abs(actual.Item2 - expected.Item2) <= tolerance,
            $"{name}: expected BGR({expected.Item0},{expected.Item1},{expected.Item2}), actual BGR({actual.Item0},{actual.Item1},{actual.Item2}), tolerance {tolerance}.");
    }
}
