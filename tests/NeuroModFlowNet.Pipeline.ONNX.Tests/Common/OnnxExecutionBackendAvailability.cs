using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class OnnxExecutionBackendAvailability
{
    static readonly Dictionary<InferenceBackend, Lazy<string?>> AvailabilityErrors = new()
    {
        [InferenceBackend.Cpu] = new Lazy<string?>(() => TryCreateIdentityModel(InferenceBackend.Cpu)),
        [InferenceBackend.Cuda] = new Lazy<string?>(() => TryCreateIdentityModel(InferenceBackend.Cuda)),
        [InferenceBackend.TensorRt] = new Lazy<string?>(() => TryCreateIdentityModel(InferenceBackend.TensorRt))
    };

    public static void AssertAvailable(InferenceBackend backend)
    {
        _ = PipelineOnnxTestEnvironment.Current;

        if(!AvailabilityErrors.TryGetValue(backend, out Lazy<string?>? errorLazy))
            throw new NotSupportedException($"Backend {backend} is not part of the ONNX operation test matrix.");

        string? error = errorLazy.Value;
        if(error is not null)
            Assert.Fail($"Backend {backend} is enabled but unavailable. Configure CUDA/cuDNN/TensorRT paths or disable this backend. Error: {error}");
    }

    static string? TryCreateIdentityModel(InferenceBackend backend)
    {
        try
        {
            byte[] modelBytes = IdentityBuilder.Build(global::Onnx.TensorProto.Types.DataType.Float, 1, 1, 1, 1);
            using var model = new OnnxModel(modelBytes, backend, displayName: $"backend-probe-{backend}");
            return null;
        }
        catch(Exception exception)
        {
            return exception.Message;
        }
    }
}
