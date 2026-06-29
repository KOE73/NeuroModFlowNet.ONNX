using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Services;

public sealed class OrtValueBatchedInferenceEndpointTests
{
    [Fact]
    public async Task OrtValueEndpoint_ConcatsRequestsAndReturnsResultArrayPerRun()
    {
        string modelPath = CreateIdentityModelPath(batchSize: 2);

        await using var endpoint = new OrtValueBatchedInferenceEndpoint<float>(
            name: "test.ortValue.identity",
            modelPath: modelPath,
            executionBackend: InferenceBackend.Cpu,
            options: new OnnxBatchedResourceOptions(MaxBatchSize: 2, MaxWaitTime: TimeSpan.FromMilliseconds(250), MaxPendingRequests: 4),
            batchInputAssembler: new ConcatOrtValueBatchInputAssembler(),
            outputShapeResolver: new PrimaryOutputShapeResolver(),
            outputDecoder: new FirstScalarPerBatchDecoder());

        var instruction = new Model_OrtValueInference<float>(
            name: "Model_OrtValueInference_Test",
            inputKey: "input",
            outputKey: "output",
            endpoint: endpoint);

        using OrtValue firstInput = CreateScalarTensor(11);
        using OrtValue secondInput = CreateScalarTensor(22);
        await using VmRunContext firstContext = VmRunContextFactory.Create();
        await using VmRunContext secondContext = VmRunContextFactory.Create();
        firstContext.Set("input", firstInput);
        secondContext.Set("input", secondInput);

        ValueTask<OpResult> firstTask = instruction.ExecuteAsync(firstContext, CancellationToken.None);
        ValueTask<OpResult> secondTask = instruction.ExecuteAsync(secondContext, CancellationToken.None);

        Assert.Equal(OpResult.Continue, await firstTask);
        Assert.Equal(OpResult.Continue, await secondTask);

        Assert.Equal([11], firstContext.Get<float[]>("output"));
        Assert.Equal([22], secondContext.Get<float[]>("output"));
    }

    static string CreateIdentityModelPath(int batchSize)
    {
        byte[] modelBytes = IdentityBuilder.Build(global::Onnx.TensorProto.Types.DataType.Float, batchSize, 1, 1, 1);
        string modelPath = Path.Combine(Path.GetTempPath(), $"nmfn-ortvalue-identity-{batchSize}-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(modelPath, modelBytes);
        return modelPath;
    }

    static OrtValue CreateScalarTensor(float value)
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [1, 1, 1, 1]);

        tensor.GetTensorMutableDataAsSpan<float>()[0] = value;
        return tensor;
    }

    sealed class FirstScalarPerBatchDecoder : IOrtValueBatchOutputDecoder<float>
    {
        public IReadOnlyList<float[]> Decode(
            OrtValue output,
            OnnxModel model,
            string outputName,
            int requestCount)
        {
            ReadOnlySpan<float> data = output.GetTensorDataAsSpan<float>();
            var results = new float[requestCount][];

            for(int index = 0; index < requestCount; index++)
                results[index] = [data[index]];

            return results;
        }
    }
}
