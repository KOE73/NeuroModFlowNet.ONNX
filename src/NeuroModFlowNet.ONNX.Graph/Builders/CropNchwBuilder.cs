using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class CropNchwBuilder
{
    public const string InputName = "image";
    public const string OutputName = "crop";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int x1,
        int y1,
        int x2,
        int y2,
        int channels,
        TensorProto.Types.DataType elementType)
    {
        if(x2 <= x1)
            throw new ArgumentOutOfRangeException(nameof(x2), "Crop x2 must be greater than x1.");

        if(y2 <= y1)
            throw new ArgumentOutOfRangeException(nameof(y2), "Crop y2 must be greater than y1.");

        int cropWidth = x2 - x1;
        int cropHeight = y2 - y1;
        var graph = new GraphProto { Name = "crop_nchw_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            elementType,
            1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(TensorInfo(
            OutputName,
            elementType,
            1, channels, cropHeight, cropWidth));

        graph.Initializer.Add(Int64Tensor("starts", [y1, x1]));
        graph.Initializer.Add(Int64Tensor("ends", [y2, x2]));
        graph.Initializer.Add(Int64Tensor("axes", [2, 3]));
        graph.Initializer.Add(Int64Tensor("steps", [1, 1]));

        graph.Node.Add(new NodeProto
        {
            Name = "crop",
            OpType = "Slice",
            Input = { InputName, "starts", "ends", "axes", "steps" },
            Output = { OutputName }
        });

        return CreateModel("neuromodflownet-crop-nchw-builder", graph).ToByteArray();
    }
}
