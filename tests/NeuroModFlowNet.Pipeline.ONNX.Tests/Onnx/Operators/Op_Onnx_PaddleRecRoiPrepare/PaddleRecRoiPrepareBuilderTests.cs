using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_PaddleRecRoiPrepare;

public sealed class PaddleRecRoiPrepareBuilderTests
{
    public enum PaddleRecRoiPrepareBuilderVariant
    {
        PerRegionGridSampleConcat,
        BatchedGridSample,
        BatchedGridSampleFromMatrices
    }

    [Theory]
    [InlineData(PaddleRecRoiPrepareBuilderVariant.PerRegionGridSampleConcat)]
    [InlineData(PaddleRecRoiPrepareBuilderVariant.BatchedGridSample)]
    [InlineData(PaddleRecRoiPrepareBuilderVariant.BatchedGridSampleFromMatrices)]
    public void BuildFP32Nchw_ProducesPaddleNormalizedBatch(PaddleRecRoiPrepareBuilderVariant variant)
    {
        const int sourceWidth = 16;
        const int sourceHeight = 8;
        const int channels = 3;
        const int targetWidth = sourceWidth;
        const int targetHeight = sourceHeight;
        const int regionCount = 2;

        IReadOnlyList<float[]> matrices =
        [
            IdentityTargetToSourceMatrix(),
            IdentityTargetToSourceMatrix()
        ];

        byte[] modelBytes = variant switch
        {
            PaddleRecRoiPrepareBuilderVariant.PerRegionGridSampleConcat =>
                PaddleRecRoiPrepareBuilder.BuildPerRegionGridSampleConcatFP32Nchw(
                    sourceWidth,
                    sourceHeight,
                    channels,
                    targetWidth,
                    targetHeight,
                    matrices),
            PaddleRecRoiPrepareBuilderVariant.BatchedGridSample =>
                PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFP32Nchw(
                    sourceWidth,
                    sourceHeight,
                    channels,
                    targetWidth,
                    targetHeight,
                    matrices),
            PaddleRecRoiPrepareBuilderVariant.BatchedGridSampleFromMatrices =>
                PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFromMatricesFP32Nchw(
                    sourceWidth,
                    sourceHeight,
                    channels,
                    targetWidth,
                    targetHeight,
                    regionCount),
            _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, null)
        };

        using var context = new OnnxExecutionContext(new OnnxModel(
            modelBytes,
            InferenceBackend.Cpu,
            displayName: $"{variant}.onnx"), ownsModel: true);

        OrtValue input = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [1, channels, sourceHeight, sourceWidth]);
        input.GetTensorMutableDataAsSpan<float>().Fill(0.5f);

        context.SetInput(PaddleRecRoiPrepareBuilder.InputName, input);
        if(variant == PaddleRecRoiPrepareBuilderVariant.BatchedGridSampleFromMatrices)
            context.SetInput(PaddleRecRoiPrepareBuilder.MatrixInputName, CreateMatrixInput(matrices));

        context.Run();

        OrtValue output = context.GetOutputValue(PaddleRecRoiPrepareBuilder.OutputName);
        var outputInfo = output.GetTensorTypeAndShape();
        Assert.Equal(TensorElementType.Float, outputInfo.ElementDataType);
        Assert.Equal((long[])[regionCount, channels, targetHeight, targetWidth], outputInfo.Shape);

        ReadOnlySpan<float> outputData = output.GetTensorDataAsSpan<float>();
        for(int index = 0; index < outputData.Length; index++)
            Assert.InRange(outputData[index], -0.00001f, 0.00001f);
    }

    static float[] IdentityTargetToSourceMatrix() =>
    [
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    ];

    static OrtValue CreateMatrixInput(IReadOnlyList<float[]> matrices)
    {
        OrtValue input = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [matrices.Count, 9]);

        Span<float> data = input.GetTensorMutableDataAsSpan<float>();
        for(int matrixIndex = 0; matrixIndex < matrices.Count; matrixIndex++)
            matrices[matrixIndex].AsSpan(0, 9).CopyTo(data.Slice(matrixIndex * 9, 9));

        return input;
    }
}
