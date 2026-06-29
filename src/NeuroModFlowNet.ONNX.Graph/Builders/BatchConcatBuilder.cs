using Google.Protobuf;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// Builds a minimal ONNX graph that concatenates fixed-shape tensors along the batch axis.
/// </summary>
public static class BatchConcatBuilder
{
    public const string OutputName = "batch";

    public static string InputName(int slotIndex) => $"input_{slotIndex}";

    public static byte[] Build(
        int batchSize,
        IReadOnlyList<long> singleInputShape,
        TensorProto.Types.DataType elementType)
    {
        if(batchSize < 2)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 2.");

        ArgumentNullException.ThrowIfNull(singleInputShape);

        if(singleInputShape.Count == 0)
            throw new ArgumentException("Single input shape must contain at least the batch dimension.", nameof(singleInputShape));

        if(singleInputShape[0] != 1)
            throw new ArgumentException($"Single input batch dimension must be 1, actual: {singleInputShape[0]}.", nameof(singleInputShape));

        long[] outputShape = [batchSize, .. singleInputShape.Skip(1)];
        var graph = new GraphProto { Name = "batch_concat" };

        for(int slotIndex = 0; slotIndex < batchSize; slotIndex++)
            graph.Input.Add(TensorInfo(InputName(slotIndex), elementType, [.. singleInputShape]));

        graph.Output.Add(TensorInfo(OutputName, elementType, outputShape));

        var concatNode = new NodeProto
        {
            Name = "concat_batch",
            OpType = "Concat",
            Output = { OutputName }
        };
        concatNode.Attribute.Add(IntAttribute("axis", 0));

        for(int slotIndex = 0; slotIndex < batchSize; slotIndex++)
            concatNode.Input.Add(InputName(slotIndex));

        graph.Node.Add(concatNode);

        return CreateModel("neuromodflownet-batch-concat-builder", graph).ToByteArray();
    }
}
