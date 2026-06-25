using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Resize;

public sealed class Op_Onnx_ResizeTests
{
    public static IEnumerable<object[]> ResizeCases =>
        from backend in OnnxExecutionBackendMatrix.EnabledBackends
        from testCase in OpOnnxResizePattern.DeterministicCases().Select(static item => (OpOnnxResizePatternCase)item[0])
        select new object[] { testCase, backend };

    [Theory]
    [MemberData(nameof(ResizeCases))]
    public async Task ExecuteAsync_ReturnsExpectedPixelsAndWritesBackTransform(OpOnnxResizePatternCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using var ownedTestCase = testCase;
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(testCase.Source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize(
            "input",
            "output",
            testCase.TargetSize,
            outputTransformKey: "transform.output.toInput",
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        using Mat actual = OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);

        OpOnnxResizePattern.AssertResult(actual, testCase);
        OpOnnxResizePattern.SaveArtifacts(actual, testCase, executionBackend);

        ICoordinateBackTransform transform = context.Get<ICoordinateBackTransform>("transform.output.toInput");
        Assert.True(transform.TryMapBackward(new System.Numerics.Vector2(testCase.TargetSize.Width, testCase.TargetSize.Height), out var mappedPoint));
        Assert.Equal(testCase.Source.Width, mappedPoint.X);
        Assert.Equal(testCase.Source.Height, mappedPoint.Y);
    }

    [Fact]
    public void Descriptor_DeclaresTransformOutputWhenRequested()
    {
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Resize(
            "input",
            "output",
            new Size(8, 6),
            outputTransformKey: "transform.output.toInput",
            isFinal: true,
            executionBackend: InferenceBackend.Cpu);

        VarRequirement transformWrite = Assert.Single(
            instruction.Descriptor.Writes,
            requirement => requirement.Key == "transform.output.toInput");

        Assert.Equal(typeof(ICoordinateBackTransform), transformWrite.ValueType);
    }
}
