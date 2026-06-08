using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// Builds small ONNX preprocessing graphs from OpenCV-style BGR U8 HWC image tensors to normalized RGB NCHW tensors.
/// </summary>
/// <remarks>
/// The graph intentionally uses elementary ONNX operators instead of a custom fused node. That keeps the generated model
/// portable across ONNX Runtime execution providers and makes the VM trace match the conceptual operation being tested:
/// color order, layout and value range conversion before a standard model input.
/// </remarks>
public static class BgrU8Hwc_To_RgbNchw_Div255Builder
{
    public const string InputName = "bgr_u8_hwc";
    public const string OutputName = "rgb_nchw";

    public static byte[] BuildFP32(int width, int height) =>
        Build(width, height, TensorProto.Types.DataType.Float);

    public static byte[] BuildFP16(int width, int height) =>
        Build(width, height, TensorProto.Types.DataType.Float16);

    static byte[] Build(int width, int height, TensorProto.Types.DataType outputDataType)
    {
        if(width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");

        if(height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

        if(outputDataType is not TensorProto.Types.DataType.Float and not TensorProto.Types.DataType.Float16)
            throw new ArgumentOutOfRangeException(nameof(outputDataType), "Output data type must be FP32 or FP16.");

        var graph = new GraphProto { Name = "bgr_u8_hwc_to_rgb_nchw_div255" };

        graph.Input.Add(TensorInfo(
            InputName,
            TensorProto.Types.DataType.Uint8,
            1, height, width, 3));

        graph.Output.Add(TensorInfo(
            OutputName,
            outputDataType,
            1, 3, height, width));

        graph.Initializer.Add(Int64Tensor("rgb_channel_indices", [2, 1, 0]));
        graph.Initializer.Add(FloatTensor("div255_scale", [1f / 255f]));

        graph.Node.Add(new NodeProto
        {
            Name = "cast_u8_to_fp32",
            OpType = "Cast",
            Input = { InputName },
            Output = { "bgr_fp32_hwc" },
            Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Float) }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "bgr_to_rgb",
            OpType = "Gather",
            Input = { "bgr_fp32_hwc", "rgb_channel_indices" },
            Output = { "rgb_fp32_hwc" },
            Attribute = { IntAttribute("axis", 3) }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "hwc_to_nchw",
            OpType = "Transpose",
            Input = { "rgb_fp32_hwc" },
            Output = { "rgb_fp32_nchw" },
            Attribute = { IntsAttribute("perm", [0, 3, 1, 2]) }
        });

        string normalizedOutputName = outputDataType == TensorProto.Types.DataType.Float
            ? OutputName
            : "rgb_fp32_nchw_div255";

        graph.Node.Add(new NodeProto
        {
            Name = "div255",
            OpType = "Mul",
            Input = { "rgb_fp32_nchw", "div255_scale" },
            Output = { normalizedOutputName }
        });

        if(outputDataType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_fp32_to_fp16",
                OpType = "Cast",
                Input = { normalizedOutputName },
                Output = { OutputName },
                Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Float16) }
            });
        }

        return CreateModel("neuromodflownet-bgr-u8-hwc-to-rgb-nchw-div255-builder", graph).ToByteArray();
    }
}
