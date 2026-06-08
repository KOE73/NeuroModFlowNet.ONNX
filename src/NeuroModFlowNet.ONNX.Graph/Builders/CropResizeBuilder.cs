using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class CropResizeBuilder
{
    public const string InputName = "image";
    public const string OutputName = "resized";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int x1,
        int y1,
        int x2,
        int y2,
        int outputWidth,
        int outputHeight,
        int channels)
    {
        if(x2 <= x1)
            throw new ArgumentOutOfRangeException(nameof(x2), "Crop x2 must be greater than x1.");

        if(y2 <= y1)
            throw new ArgumentOutOfRangeException(nameof(y2), "Crop y2 must be greater than y1.");

        var graph = new GraphProto { Name = "crop_resize" };

        graph.Input.Add(TensorInfo(
            InputName,
            TensorProto.Types.DataType.Uint8,
            1, sourceHeight, sourceWidth, channels));

        graph.Output.Add(TensorInfo(
            OutputName,
            TensorProto.Types.DataType.Uint8,
            1, outputHeight, outputWidth, channels));

        graph.Initializer.Add(Int64Tensor("starts", [y1, x1]));
        graph.Initializer.Add(Int64Tensor("ends", [y2, x2]));
        graph.Initializer.Add(Int64Tensor("axes", [1, 2]));
        graph.Initializer.Add(Int64Tensor("steps", [1, 1]));
        graph.Initializer.Add(FloatTensor("roi", [0, 0, 0, 0, 1, 1, 1, 1]));
        graph.Initializer.Add(FloatTensor(
            "scales",
            [
                1,
                (float)outputHeight / (y2 - y1),
                (float)outputWidth / (x2 - x1),
                1
            ]));

        graph.Node.Add(new NodeProto
        {
            Name = "crop",
            OpType = "Slice",
            Input = { InputName, "starts", "ends", "axes", "steps" },
            Output = { "crop" }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { "crop", "roi", "scales" },
            Output = { OutputName },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "half_pixel"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        return CreateModel("neuromodflownet-runtime-crop-resize-builder", graph).ToByteArray();
    }
}
