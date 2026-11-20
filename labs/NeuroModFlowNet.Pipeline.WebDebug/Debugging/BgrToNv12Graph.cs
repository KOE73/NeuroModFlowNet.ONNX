using Google.Protobuf;
using Onnx;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal static class BgrToNv12Graph
{
    public static byte[] Build(int width, int height, int pitch)
    {
        var graph = new GraphProto { Name = "debug_bgr_to_nv12_bt709_limited" };
        var u8 = TensorProto.Types.DataType.Uint8;
        graph.Input.Add(TensorInfo("bgr", u8, 1, height, width, 3));
        graph.Output.Add(TensorInfo("y", u8, height, pitch));
        graph.Output.Add(TensorInfo("uv", u8, height / 2, pitch));
        Add("Cast", ["bgr"], "bgr_float", IntAttribute("to", (long)TensorProto.Types.DataType.Float));
        Add("Transpose", ["bgr_float"], "nchw", IntsAttribute("perm", [0, 3, 1, 2]));
        Add("AveragePool", ["nchw"], "half", IntsAttribute("kernel_shape", [2, 2]), IntsAttribute("strides", [2, 2]));

        const float blue = .0722f, green = .7152f, red = .2126f;
        const float luma = 219f / 255, chroma = 224f / 255;
        Plane("y", "nchw", height, [blue * luma, green * luma, red * luma], [16.5f]);
        Plane("uv", "half", height / 2,
            [.5f * chroma, -.5f * green / (1 - blue) * chroma, -.5f * red / (1 - blue) * chroma,
             -.5f * blue / (1 - red) * chroma, -.5f * green / (1 - red) * chroma, .5f * chroma], [128.5f, 128.5f]);
        return CreateModel("debug-bgr-to-nv12", graph).ToByteArray();

        void Plane(string name, string input, int rows, float[] weights, float[] offsets)
        {
            TensorProto matrix = FloatTensor(name + "_weights", weights);
            matrix.Dims.Clear();
            matrix.Dims.AddRange([offsets.Length, 3, 1, 1]);
            graph.Initializer.Add(matrix);
            graph.Initializer.Add(FloatTensor(name + "_offsets", offsets));
            graph.Initializer.Add(Int64Tensor(name + "_shape", [rows, width]));
            Add("Conv", [input, matrix.Name, name + "_offsets"], name + "_channels");
            Add("Transpose", [name + "_channels"], name + "_packed", IntsAttribute("perm", [0, 2, 3, 1]));
            Add("Reshape", [name + "_packed", name + "_shape"], name + "_dense");
            string packed = name + "_dense";
            if(pitch != width)
            {
                graph.Initializer.Add(Int64Tensor(name + "_pads", [0, 0, 0, pitch - width]));
                Add("Pad", [packed, name + "_pads"], name + "_pitched");
                packed = name + "_pitched";
            }
            Add("Cast", [packed], name, IntAttribute("to", (long)u8));
        }

        void Add(string op, string[] inputs, string output, params AttributeProto[] attributes)
        {
            var node = new NodeProto { Name = output, OpType = op };
            node.Input.AddRange(inputs);
            node.Output.Add(output);
            node.Attribute.AddRange(attributes);
            graph.Node.Add(node);
        }
    }
}
