using Google.Protobuf;
using Onnx;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// EN: Packs opaque BGRA U8 NHWC from BGR U8 NHWC for a GPU render target.
///
/// RU: Упаковывает BGR U8 NHWC в непрозрачный BGRA U8 NHWC для GPU-рендеринга.
/// </summary>
public static class BgrToBgraBuilder
{
    public static byte[] Build(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        const TensorProto.Types.DataType elementType = TensorProto.Types.DataType.Uint8;
        var graph = new GraphProto { Name = "bgr_u8_nhwc_to_bgra_u8_nhwc" };
        graph.Input.Add(TensorInfo("bgr", elementType, 1, height, width, 3));
        graph.Output.Add(TensorInfo("bgra", elementType, 1, height, width, 4));
        byte[] alpha = new byte[checked(width * height)];
        alpha.AsSpan().Fill(255);
        var opaque = new TensorProto { Name = "opaque", DataType = (int)elementType, RawData = ByteString.CopyFrom(alpha) };
        opaque.Dims.AddRange([1, height, width, 1]);
        graph.Initializer.Add(opaque);
        var pack = new NodeProto { OpType = "Concat" };
        pack.Input.AddRange(["bgr", "opaque"]);
        pack.Output.Add("bgra");
        pack.Attribute.Add(IntAttribute("axis", 3));
        graph.Node.Add(pack);
        return CreateModel(graph.Name, graph).ToByteArray();
    }
}
