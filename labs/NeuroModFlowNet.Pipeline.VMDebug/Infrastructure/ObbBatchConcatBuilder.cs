using Google.Protobuf;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Builds a minimal ONNX graph that concatenates <paramref name="batchSize"/> identical-shape tensors along axis 0.
/// </summary>
/// <remarks>
/// The produced model accepts <paramref name="batchSize"/> inputs each shaped [1, H, W, C] and outputs one tensor
/// shaped [batchSize, H, W, C]. This concat step feeds a full batch to the OBB model without staging data through
/// host memory: if the inputs live in device memory and CUDA is selected, the Concat graph runs on-device.
///
/// Why Concat and not Sequence/ConcatFromSequence:
///   Sequence types require a different C# API and ORT gives less optimization leverage for CUDA/TensorRT.
///   A plain Concat over a fixed number of named inputs is the standard batch-assembly path for YOLO.
/// </remarks>
internal static class ObbBatchConcatBuilder
{
    public const string OutputName = "batch";

    public static string InputName(int slotIndex) => $"input_{slotIndex}";

    /// <summary>
    /// Builds the serialized ONNX model bytes for a Concat-batch graph.
    /// </summary>
    /// <param name="batchSize">Number of input tensors (== VM count).</param>
    /// <param name="height">Spatial height of each input.</param>
    /// <param name="width">Spatial width of each input.</param>
    /// <param name="channels">Channel count of each input.</param>
    /// <param name="elementType">Element data type (e.g. Uint8 for BGR camera frames).</param>
    public static byte[] Build(
        int batchSize,
        long height,
        long width,
        long channels,
        TensorProto.Types.DataType elementType)
    {
        if(batchSize < 2)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 2.");

        var graph = new GraphProto { Name = "obb_batch_concat" };

        for(int slotIndex = 0; slotIndex < batchSize; slotIndex++)
            graph.Input.Add(TensorInfo(InputName(slotIndex), elementType, 1, height, width, channels));

        graph.Output.Add(TensorInfo(OutputName, elementType, batchSize, height, width, channels));

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

        return CreateModel("neuromodflownet-obb-batch-concat", graph).ToByteArray();
    }
}
