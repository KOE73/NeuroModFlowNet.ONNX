using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Rotate90;

public sealed class Op_Onnx_Rotate90Tests
{
    public static IEnumerable<object[]> RotateCases =>
        from backend in OnnxExecutionBackendMatrix.EnabledBackends
        from testCase in OpOnnxRotate90Pattern.DeterministicCases().Select(static item => (OpOnnxRotate90PatternCase)item[0])
        select new object[] { testCase, backend };

    [Theory]
    [MemberData(nameof(RotateCases))]
    public async Task ExecuteAsync_ReturnsExpectedPixels(OpOnnxRotate90PatternCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using var ownedTestCase = testCase;
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(testCase.Source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_U8_NHWC(
            "input",
            "output",
            testCase.Mode,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        using Mat actual = OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);

        OpOnnxRotate90Pattern.AssertResult(actual, testCase);
        OpOnnxRotate90Pattern.SaveArtifacts(actual, testCase, executionBackend);
    }

    [Fact]
    public async Task ExecuteAsync_WritesBackTransform()
    {
        OnnxExecutionBackendAvailability.AssertAvailable(InferenceBackend.Cpu);

        using var source = OpOnnxRotate90Pattern.CreateCornerMarkerCase(960, 640, Rotate90Mode.Clockwise90);
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(source.Source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Rotate90_U8_NHWC(
            "input",
            "output",
            Rotate90Mode.Clockwise90,
            outputTransformKey: "rotateToSource",
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        ICoordinateBackTransform transform = context.Get<ICoordinateBackTransform>("rotateToSource");
        bool mapped = transform.TryMapBackward(new Vector2(0, 0), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(0, 639), point);
    }
}
