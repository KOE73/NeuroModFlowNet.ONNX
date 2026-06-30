using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Runtime;

public sealed class RuntimeOnnxOperatorKernelCacheTests
{
    [Fact]
    public void CreateContext_SameKey_ReusesModelButKeepsExecutionContextSeparate()
    {
        RuntimeOnnxOperatorKernelKey key = new(
            "test.identity",
            "shape=1x1;type=float",
            InferenceBackend.Cpu,
            "cpu");

        using OnnxExecutionContext first = RuntimeOnnxOperatorKernelCache.CreateContext(
            key,
            () => IdentityBuilder.Build(global::Onnx.TensorProto.Types.DataType.Float, 1, 1),
            configure: null,
            "test-identity.onnx");

        using OnnxExecutionContext second = RuntimeOnnxOperatorKernelCache.CreateContext(
            key,
            () => throw new InvalidOperationException("The model must already be cached for this key."),
            configure: null,
            "test-identity.onnx");

        Assert.NotSame(first, second);
        Assert.Same(first.Model, second.Model);
        Assert.NotSame(first.IoBinding, second.IoBinding);
        Assert.NotSame(first.RunOptions, second.RunOptions);
    }

    [Fact]
    public void CreateContext_DifferentSemanticKey_DoesNotReuseModel()
    {
        RuntimeOnnxOperatorKernelKey firstKey = new(
            "test.identity",
            "shape=1x1;type=float",
            InferenceBackend.Cpu,
            "cpu");

        RuntimeOnnxOperatorKernelKey secondKey = firstKey with { SemanticKey = "shape=1x2;type=float" };

        using OnnxExecutionContext first = RuntimeOnnxOperatorKernelCache.CreateContext(
            firstKey,
            () => IdentityBuilder.Build(global::Onnx.TensorProto.Types.DataType.Float, 1, 1),
            configure: null,
            "test-identity-1.onnx");

        using OnnxExecutionContext second = RuntimeOnnxOperatorKernelCache.CreateContext(
            secondKey,
            () => IdentityBuilder.Build(global::Onnx.TensorProto.Types.DataType.Float, 1, 2),
            configure: null,
            "test-identity-2.onnx");

        Assert.NotSame(first.Model, second.Model);
        Assert.Equal(TensorElementType.Float, first.Model.GetInputElementType(IdentityBuilder.InputName));
        Assert.Equal(TensorElementType.Float, second.Model.GetInputElementType(IdentityBuilder.InputName));
    }
}
