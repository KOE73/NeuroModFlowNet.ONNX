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

        graph.Initializer.Add(FloatTensor("roi", [0, 0, 0, 0, 1, 1, 1, 1]));
        graph.Initializer.Add(FloatTensor(
            "scales",
            [
                1,
                (float)outputHeight / sourceHeight,
                (float)outputWidth / sourceWidth,
                1
            ]));

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { InputName, "roi", "scales" },
            Output = { OutputName },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "half_pixel"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        return CreateModel("neuromodflownet-runtime-resize-builder", graph).ToByteArray();
    }
}
