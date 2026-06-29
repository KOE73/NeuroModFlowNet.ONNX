using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class ResizeNchwBuilder
{
    public const string InputName = "image";
    public const string OutputName = "resized";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        TensorProto.Types.DataType elementType)
    {
        if(elementType is not (TensorProto.Types.DataType.Float or TensorProto.Types.DataType.Float16))
            throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "NCHW resize currently supports FP32 and FP16.");

        var graph = new GraphProto { Name = "resize_nchw_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            elementType,
            1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(TensorInfo(
            OutputName,
            elementType,
            1, channels, outputHeight, outputWidth));

        graph.Initializer.Add(Int64Tensor("sizes", [1, channels, outputHeight, outputWidth]));

        string resizeInputName = InputName;
        if(elementType == TensorProto.Types.DataType.Float16)
        {
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
            resizeInputName = "image_float";
        }

        string resizeOutputName = elementType == TensorProto.Types.DataType.Float
            ? OutputName
            : "resized_float";

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { resizeInputName, "", "", "sizes" },
            Output = { resizeOutputName },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "asymmetric"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        if(elementType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_output_to_float16",
                OpType = "Cast",
                Input = { resizeOutputName },
                Output = { OutputName },
                Attribute =
                {
                    IntAttribute("to", (long)TensorProto.Types.DataType.Float16)
                }
            });
        }

        return CreateModel("neuromodflownet-resize-nchw-builder", graph).ToByteArray();
    }
}
