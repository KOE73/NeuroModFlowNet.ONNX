using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_PadResize;

public sealed class Op_Onnx_PadResizeTests
{
    public static IEnumerable<object[]> PadResizeCases =>
        from backend in OnnxExecutionBackendMatrix.EnabledBackends
        from testCase in OpOnnxPadResizePattern.DeterministicCases().Select(static item => (OpOnnxPadResizePatternCase)item[0])
        select new object[] { testCase, backend };

    [Theory]
    [MemberData(nameof(PadResizeCases))]
    public async Task ExecuteAsync_ReturnsExpectedPixels(OpOnnxPadResizePatternCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using var ownedTestCase = testCase;
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(testCase.Source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_U8_NHWC(
            "input",
            "output",
            testCase.TargetSize,
            stride: testCase.Stride,
            mode: testCase.Mode,
            padValue: testCase.PadValue,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        using Mat actual = OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);

        OpOnnxPadResizePattern.AssertResult(actual, testCase);
        OpOnnxPadResizePattern.SaveArtifacts(actual, testCase, executionBackend);
    }

    [Fact]
    public async Task ExecuteAsync_WritesBackTransform()
    {
        OnnxExecutionBackendAvailability.AssertAvailable(InferenceBackend.Cpu);

        using var source = new Mat(500, 1000, MatType.CV_8UC3, new Scalar(7, 149, 231));
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_PadResize_U8_NHWC(
            "input",
            "output",
            new Size(640, 640),
            outputTransformKey: "padResizeToSource",
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        var transform = Assert.IsType<PadResizeCoordinateBackTransform>(
            context.Get<ICoordinateBackTransform>("padResizeToSource"));
        Assert.Equal(640, transform.OutputWidth);
        Assert.Equal(640, transform.OutputHeight);
        Assert.Equal(160, transform.PadTop);
        Assert.Equal(0.64f, transform.Scale);
    }
}
