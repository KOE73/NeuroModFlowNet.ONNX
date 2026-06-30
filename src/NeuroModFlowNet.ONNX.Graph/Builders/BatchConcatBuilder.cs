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
        int boundInputCount,
        IReadOnlyList<long> singleInputShape,
        TensorProto.Types.DataType elementType)
    {
        if(batchSize < 2)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 2.");

        if(boundInputCount <= 0 || boundInputCount > batchSize)
            throw new ArgumentOutOfRangeException(nameof(boundInputCount), "Bound input count must be positive and not greater than batch size.");

        ArgumentNullException.ThrowIfNull(singleInputShape);

        if(singleInputShape.Count == 0)
            throw new ArgumentException("Single input shape must contain at least the batch dimension.", nameof(singleInputShape));

        if(singleInputShape[0] != 1)
            throw new ArgumentException($"Single input batch dimension must be 1, actual: {singleInputShape[0]}.", nameof(singleInputShape));

        long[] outputShape = [batchSize, .. singleInputShape.Skip(1)];
        var graph = new GraphProto { Name = "batch_concat" };

        for(int slotIndex = 0; slotIndex < boundInputCount; slotIndex++)
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
            concatNode.Input.Add(InputName(slotIndex % boundInputCount));

        graph.Node.Add(concatNode);

        return CreateModel("neuromodflownet-batch-concat-builder", graph).ToByteArray();
    }

    public static byte[] BuildVariableBatch(
        IReadOnlyList<IReadOnlyList<long>> inputShapes,
        TensorProto.Types.DataType elementType)
    {
        ArgumentNullException.ThrowIfNull(inputShapes);

        if(inputShapes.Count < 2)
            throw new ArgumentOutOfRangeException(nameof(inputShapes), "Variable batch concat requires at least two inputs.");

        IReadOnlyList<long> firstShape = inputShapes[0];
        if(firstShape.Count == 0 || firstShape[0] <= 0)
            throw new ArgumentException("Input shape must contain a positive batch dimension.", nameof(inputShapes));

        long outputBatchSize = firstShape[0];
        for(int inputIndex = 1; inputIndex < inputShapes.Count; inputIndex++)
        {
            IReadOnlyList<long> inputShape = inputShapes[inputIndex];
            if(inputShape.Count != firstShape.Count)
                throw new ArgumentException($"Input {inputIndex} rank differs from input 0.", nameof(inputShapes));

            if(inputShape[0] <= 0)
                throw new ArgumentException($"Input {inputIndex} batch dimension must be positive.", nameof(inputShapes));

            for(int dimensionIndex = 1; dimensionIndex < inputShape.Count; dimensionIndex++)
            {
                if(inputShape[dimensionIndex] != firstShape[dimensionIndex])
                    throw new ArgumentException($"Input {inputIndex} dimension {dimensionIndex} differs from input 0.", nameof(inputShapes));
            }

            outputBatchSize += inputShape[0];
        }

        long[] outputShape = [outputBatchSize, .. firstShape.Skip(1)];
        var graph = new GraphProto { Name = "variable_batch_concat" };

        for(int slotIndex = 0; slotIndex < inputShapes.Count; slotIndex++)
            graph.Input.Add(TensorInfo(InputName(slotIndex), elementType, [.. inputShapes[slotIndex]]));

        graph.Output.Add(TensorInfo(OutputName, elementType, outputShape));

        var concatNode = new NodeProto
        {
            Name = "concat_variable_batch",
            OpType = "Concat",
            Output = { OutputName }
        };
        concatNode.Attribute.Add(IntAttribute("axis", 0));

        for(int slotIndex = 0; slotIndex < inputShapes.Count; slotIndex++)
            concatNode.Input.Add(InputName(slotIndex));

        graph.Node.Add(concatNode);

        return CreateModel("neuromodflownet-variable-batch-concat-builder", graph).ToByteArray();
    }
}
