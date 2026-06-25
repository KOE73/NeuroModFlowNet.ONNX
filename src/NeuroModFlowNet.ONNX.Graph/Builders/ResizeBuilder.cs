using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class ResizeBuilder
{
    public const string InputName = "image";
    public const string OutputName = "resized";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels)
    {
        var graph = new GraphProto { Name = "resize_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            TensorProto.Types.DataType.Uint8,
            1, sourceHeight, sourceWidth, channels));

        graph.Output.Add(TensorInfo(
            OutputName,
            TensorProto.Types.DataType.Uint8,
            1, outputHeight, outputWidth, channels));

        graph.Initializer.Add(Int64Tensor("sizes", [1, channels, outputHeight, outputWidth]));

        graph.Node.Add(new NodeProto
        {
            Name = "cast_input_to_float",
            OpType = "Cast",
            Input = { InputName },
            Output = { "image_float" },
            Attribute =
            {
                IntAttribute("to", (long)TensorProto.Types.DataType.Float)
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "transpose_input_to_nchw",
            OpType = "Transpose",
            Input = { "image_float" },
            Output = { "image_nchw" },
            Attribute =
            {
                IntsAttribute("perm", [0, 3, 1, 2])
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { "image_nchw", "", "", "sizes" },
            Output = { "resized_nchw" },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "asymmetric"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "transpose_output_to_nhwc",
            OpType = "Transpose",
            Input = { "resized_nchw" },
            Output = { "resized_float" },
            Attribute =
            {
                IntsAttribute("perm", [0, 2, 3, 1])
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "cast_output_to_uint8",
            OpType = "Cast",
            Input = { "resized_float" },
            Output = { OutputName },
            Attribute =
            {
                IntAttribute("to", (long)TensorProto.Types.DataType.Uint8)
            }
        });

        return CreateModel("neuromodflownet-runtime-resize-builder", graph).ToByteArray();
    }
}
