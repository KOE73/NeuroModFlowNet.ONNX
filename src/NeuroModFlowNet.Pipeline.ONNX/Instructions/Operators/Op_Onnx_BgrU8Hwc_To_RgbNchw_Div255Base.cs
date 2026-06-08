using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Base class for BGR U8 HWC image tensor preparation commands.
/// </summary>
/// <remarks>
/// OpenCV and camera frames naturally arrive as BGR bytes in HWC layout. Standard vision models usually expect RGB,
/// normalized, channel-first tensors. This base fixes the shared contract and leaves only the final floating-point
/// precision to concrete instructions.
/// </remarks>
public abstract class Op_Onnx_BgrU8Hwc_To_RgbNchw_Div255Base : Op_Onnx_TensorTransformBase
{
    protected Op_Onnx_BgrU8Hwc_To_RgbNchw_Div255Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        bool isFinal)
        : base(descriptor, inputKey, outputKey, isFinal)
    {
    }

    protected override string GraphInputName => BgrU8Hwc_To_RgbNchw_Div255Builder.InputName;

    protected override string GraphOutputName => BgrU8Hwc_To_RgbNchw_Div255Builder.OutputName;

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 BGR image data, actual element type: {inputElementType}.";

        return ValidateNhwcImageShape(inputShape, expectedChannels: 3);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, 3, inputShape[1], inputShape[2]];
}
