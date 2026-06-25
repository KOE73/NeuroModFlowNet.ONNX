using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Converts a BGR UInt8 NHWC image tensor into an RGB FP32 NCHW tensor normalized by 1/255.
/// </summary>
/// <remarks>
/// This is the standard preparation step for FP32 models that expect channel-first RGB float input in the [0, 1] range.
/// The command is intentionally explicit instead of parameterized: the VM trace should show the exact source format,
/// target precision, target layout and normalization rule.
/// </remarks>
public sealed class Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255 : Op_Onnx_BgrU8Hwc_To_RgbNchw_Div255Base
{
    public Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
        string inputKey,
        string outputKey,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            OpDescriptor.Create(
                "Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255",
                "op.onnx.bgrU8Hwc.toRgbFP32Nchw.div255",
                reads: [VarRequirement.Read<OrtValue>(inputKey)],
                writes: [VarRequirement.Write<OrtValue>(outputKey)]),
            inputKey,
            outputKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "bgr-u8-hwc-to-rgb-fp32-nchw-div255.onnx";

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType) =>
        BgrU8Hwc_To_RgbNchw_Div255Builder.BuildFP32(
            width: checked((int)inputShape[2]),
            height: checked((int)inputShape[1]));

    protected override TensorElementType GetOutputElementType(TensorElementType inputElementType) =>
        TensorElementType.Float;
}
