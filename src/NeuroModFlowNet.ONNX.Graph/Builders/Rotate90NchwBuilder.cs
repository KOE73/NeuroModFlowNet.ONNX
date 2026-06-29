using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class Rotate90NchwBuilder
{
    public const string InputName = "image";
    public const string OutputName = "rotated";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int clockwiseQuarterTurns,
        TensorProto.Types.DataType elementType)
    {
        int normalizedTurns = ((clockwiseQuarterTurns % 4) + 4) % 4;
        if(normalizedTurns == 0)
            throw new ArgumentOutOfRangeException(nameof(clockwiseQuarterTurns), "Rotate90 builder requires 90, 180, or 270 degrees.");

        int outputWidth = normalizedTurns == 2 ? sourceWidth : sourceHeight;
        int outputHeight = normalizedTurns == 2 ? sourceHeight : sourceWidth;
        var graph = new GraphProto { Name = "rotate90_nchw_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            elementType,
            1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(TensorInfo(
            OutputName,
            elementType,
            1, channels, outputHeight, outputWidth));

        AddReverseAxisInitializers(graph, "reverse_height", sourceHeight, axis: 2);
        AddReverseAxisInitializers(graph, "reverse_width", sourceWidth, axis: 3);
        AddReverseAxisInitializers(graph, "reverse_transposed_height", sourceWidth, axis: 2);
        AddReverseAxisInitializers(graph, "reverse_transposed_width", sourceHeight, axis: 3);

        if(normalizedTurns == 1)
        {
            AddTranspose(graph, InputName, "transposed", [0, 1, 3, 2]);
            AddSlice(graph, "reverse_transposed_width", "transposed", OutputName);
        }
        else if(normalizedTurns == 2)
        {
            AddSlice(graph, "reverse_height", InputName, "height_reversed");
            AddSlice(graph, "reverse_width", "height_reversed", OutputName);
        }
        else
        {
            AddTranspose(graph, InputName, "transposed", [0, 1, 3, 2]);
            AddSlice(graph, "reverse_transposed_height", "transposed", OutputName);
        }

        return CreateModel("neuromodflownet-rotate90-nchw-builder", graph).ToByteArray();
    }

    static void AddReverseAxisInitializers(GraphProto graph, string prefix, int length, long axis)
    {
        graph.Initializer.Add(Int64Tensor($"{prefix}_starts", [length - 1]));
        graph.Initializer.Add(Int64Tensor($"{prefix}_ends", [-length - 1]));
        graph.Initializer.Add(Int64Tensor($"{prefix}_axes", [axis]));
        graph.Initializer.Add(Int64Tensor($"{prefix}_steps", [-1]));
    }

    static void AddTranspose(GraphProto graph, string inputName, string outputName, ReadOnlySpan<long> permutation)
    {
        graph.Node.Add(new NodeProto
        {
            Name = outputName,
            OpType = "Transpose",
            Input = { inputName },
            Output = { outputName },
            Attribute =
            {
                IntsAttribute("perm", permutation)
            }
        });
    }

    static void AddSlice(GraphProto graph, string prefix, string inputName, string outputName)
    {
        graph.Node.Add(new NodeProto
        {
            Name = outputName,
            OpType = "Slice",
            Input = { inputName, $"{prefix}_starts", $"{prefix}_ends", $"{prefix}_axes", $"{prefix}_steps" },
            Output = { outputName }
        });
    }
}
