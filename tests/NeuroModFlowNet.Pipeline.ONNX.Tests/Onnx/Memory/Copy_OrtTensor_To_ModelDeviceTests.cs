using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Memory;

public sealed class Copy_OrtTensor_To_ModelDeviceTests
{
    [Fact]
    public async Task CopyOrtTensorToModelDevice_GpuBackendsAllocateOutputOnModelDevice()
    {
        foreach(InferenceBackend executionBackend in OnnxExecutionBackendMatrix.EnabledBackends.Where(static backend => backend != InferenceBackend.Cpu))
        {
            OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

            using OrtValue input = CreateFP32Tensor();
            await using var context = VmRunContextFactory.Create();
            using var instruction = new Copy_OrtTensor_To_ModelDevice("input", "output", executionBackend);

            context.Set("input", input);

            OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

            Assert.Equal(OpResultKind.Continue, result.Kind);

            OrtValue output = context.Get<OrtValue>("output");
            Assert.Equal(TensorElementType.Float, output.GetTensorTypeAndShape().ElementDataType);
            Assert.Equal([1, 1, 1, 1], output.GetTensorTypeAndShape().Shape);
            OrtValueMemoryAssert.AssertGpuTensor(output, executionBackend);
        }
    }

    static OrtValue CreateFP32Tensor()
    {
        OrtValue tensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [1, 1, 1, 1]);

        tensor.GetTensorMutableDataAsSpan<float>()[0] = 1;
        return tensor;
    }
}
