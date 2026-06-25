using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Crop;

public sealed class Op_Onnx_CropTests
{
    public static IEnumerable<object[]> CropCases =>
        from backend in OnnxExecutionBackendMatrix.EnabledBackends
        from testCase in OpOnnxCropPattern.DeterministicCases().Select(static item => (OpOnnxCropPatternCase)item[0])
        select new object[] { testCase, backend };

    [Theory]
    [MemberData(nameof(CropCases))]
    public async Task ExecuteAsync_ReturnsExpectedPixels(OpOnnxCropPatternCase testCase, InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using var ownedTestCase = testCase;
        using OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(testCase.Source);
        await using var context = VmRunContextFactory.Create();
        using var instruction = new global::NeuroModFlowNet.Pipeline.ONNX.Op_Onnx_Crop(
            "input",
            "output",
            testCase.CropRect,
            isFinal: true,
            executionBackend: executionBackend);

        context.Set("input", input);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        OrtValue output = context.Get<OrtValue>("output");
        using Mat actual = OrtTestTensorFactory.CreateBgrMatFromNhwcTensor(output);

        OpOnnxCropPattern.AssertResult(actual, testCase);
        OpOnnxCropPattern.SaveArtifacts(actual, testCase, executionBackend);
    }
}
