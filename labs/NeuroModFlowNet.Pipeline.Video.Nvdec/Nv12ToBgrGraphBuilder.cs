using NeuroModFlowNet.ONNX.Graph.Builders;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: Builds an ONNX graph <c>NV12 [rows, pitch] U8</c> (pitched device surface, Y plane of <c>allocHeight</c> rows
/// followed by interleaved UV) → <c>BGR [1, H, W, 3] U8 NHWC</c>. Chroma is upsampled with nearest neighbour; the colour
/// matrix and range offsets are folded into one 1x1 convolution, so the whole conversion is a handful of kernels.
///
/// RU: Строит ONNX-граф <c>NV12 [rows, pitch] U8</c> (поверхность с pitch, плоскость Y из <c>allocHeight</c> строк, затем
/// чередующиеся UV) → <c>BGR [1, H, W, 3] U8 NHWC</c>. Цветность увеличивается nearest; матрица и смещения диапазона
/// свёрнуты в одну свёртку 1x1, поэтому вся конвертация это несколько ядер.
/// </summary>
public static class Nv12ToBgrGraphBuilder
{
    public const string InputName = "nv12";
    public const string OutputName = "bgr";

    public static byte[] Build(int rows, int pitch, int width, int height, Nv12ColorMatrix matrix)
    {
        if(rows % 3 != 0)
            throw new ArgumentException($"NV12 tensor rows {rows} must be allocHeight * 3 / 2.", nameof(rows));

        int allocatedHeight = rows / 3 * 2;
        if(width > pitch || height > allocatedHeight || width % 2 != 0 || height % 2 != 0)
            throw new ArgumentException($"Frame {width}x{height} does not fit NV12 surface pitch {pitch}, alloc height {allocatedHeight}.");

        var graph = new GraphProto { Name = "nv12_to_bgr_u8_nhwc" };
        graph.Input.Add(TensorInfo(InputName, TensorProto.Types.DataType.Uint8, rows, pitch));
        graph.Output.Add(TensorInfo(OutputName, TensorProto.Types.DataType.Uint8, 1, height, width, 3));

        graph.Initializer.Add(Int64Tensor("axes_01", [0, 1]));
        graph.Initializer.Add(Int64Tensor("y_starts", [0, 0]));
        graph.Initializer.Add(Int64Tensor("y_ends", [height, width]));
        graph.Initializer.Add(Int64Tensor("uv_starts", [allocatedHeight, 0]));
        graph.Initializer.Add(Int64Tensor("uv_ends", [allocatedHeight + (height / 2), width]));
        graph.Initializer.Add(Int64Tensor("y_shape", [1, 1, height, width]));
        graph.Initializer.Add(Int64Tensor("uv_shape", [height / 2, width / 2, 2]));
        graph.Initializer.Add(FloatTensor("uv_scales", [1f, 1f, 2f, 2f]));
        graph.Initializer.Add(FloatTensor("round_half", [0.5f]));
        graph.Initializer.Add(FloatTensor("clip_min", [0f]));
        graph.Initializer.Add(FloatTensor("clip_max", [255f]));

        (float[] weights, float[] bias) = CreateConvolution(matrix);
        graph.Initializer.Add(ShapedFloatTensor("yuv_to_bgr_w", [3, 3, 1, 1], weights));
        graph.Initializer.Add(FloatTensor("yuv_to_bgr_b", bias));

        // TensorRT accepts UINT8 only at graph boundaries (network input / output). The surface is cast to float
        // first and the result is cast back to UINT8 as the very last node, so every intermediate tensor is float and
        // the whole graph stays inside one TensorRT engine instead of being partitioned to the CUDA EP.
        Add(graph, "Cast", ["nv12"], ["nv12_f"], IntAttribute("to", (long)TensorProto.Types.DataType.Float));

        // Y: [H, W] -> [1, 1, H, W]
        Add(graph, "Slice", ["nv12_f", "y_starts", "y_ends", "axes_01"], ["y_f"]);
        Add(graph, "Reshape", ["y_f", "y_shape"], ["y_nchw"]);

        // UV: [H/2, W] interleaved -> [H/2, W/2, 2] -> [1, 2, H/2, W/2] -> nearest x2 -> [1, 2, H, W]
        Add(graph, "Slice", ["nv12_f", "uv_starts", "uv_ends", "axes_01"], ["uv_f"]);
        Add(graph, "Reshape", ["uv_f", "uv_shape"], ["uv_hw2"]);
        Add(graph, "Transpose", ["uv_hw2"], ["uv_2hw"], IntsAttribute("perm", [2, 0, 1]));
        Add(graph, "Unsqueeze", ["uv_2hw", "axis_0"], ["uv_half"]);
        graph.Initializer.Add(Int64Tensor("axis_0", [0]));
        Add(graph, "Resize", ["uv_half", "", "uv_scales"], ["uv_full"],
            StringAttribute("mode", "nearest"),
            StringAttribute("nearest_mode", "floor"),
            StringAttribute("coordinate_transformation_mode", "asymmetric"));

        // [Y, U, V] -> 1x1 conv (matrix + range offsets) -> BGR float
        Add(graph, "Concat", ["y_nchw", "uv_full"], ["yuv"], IntAttribute("axis", 1));
        Add(graph, "Conv", ["yuv", "yuv_to_bgr_w", "yuv_to_bgr_b"], ["bgr_f"]);
        Add(graph, "Add", ["bgr_f", "round_half"], ["bgr_rounded"]);
        Add(graph, "Clip", ["bgr_rounded", "clip_min", "clip_max"], ["bgr_clipped"]);
        Add(graph, "Transpose", ["bgr_clipped"], ["bgr_nhwc_f"], IntsAttribute("perm", [0, 2, 3, 1]));
        Add(graph, "Cast", ["bgr_nhwc_f"], [OutputName], IntAttribute("to", (long)TensorProto.Types.DataType.Uint8));

        return CreateModel("neuromodflownet-nv12-to-bgr-u8-nhwc", graph).ToByteArray();
    }

    /// <summary>
    /// Output channels B, G, R from raw input channels Y, U, V (0..255):
    /// out = M * (s_y (Y - y0), s_c (U - 128), s_c (V - 128)).
    /// </summary>
    static (float[] Weights, float[] Bias) CreateConvolution(Nv12ColorMatrix matrix)
    {
        (double kr, double kb) = matrix is Nv12ColorMatrix.Bt709Full or Nv12ColorMatrix.Bt709Limited
            ? (0.2126, 0.0722)
            : (0.299, 0.114);
        double kg = 1.0 - kr - kb;

        bool limited = matrix is Nv12ColorMatrix.Bt601Limited or Nv12ColorMatrix.Bt709Limited;
        double lumaScale = limited ? 255.0 / 219.0 : 1.0;
        double lumaOffset = limited ? 16.0 : 0.0;
        double chromaScale = limited ? 255.0 / 224.0 : 1.0;

        // Rows: B, G, R. Columns: luma, U, V coefficients applied to centred/scaled components.
        double[,] rgbFromYuv =
        {
            { 1.0, 2.0 * (1.0 - kb), 0.0 },
            { 1.0, -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg },
            { 1.0, 0.0, 2.0 * (1.0 - kr) }
        };

        var weights = new float[9];
        var bias = new float[3];
        for(int output = 0; output < 3; output++)
        {
            double luma = rgbFromYuv[output, 0] * lumaScale;
            double u = rgbFromYuv[output, 1] * chromaScale;
            double v = rgbFromYuv[output, 2] * chromaScale;
            weights[(output * 3) + 0] = (float)luma;
            weights[(output * 3) + 1] = (float)u;
            weights[(output * 3) + 2] = (float)v;
            bias[output] = (float)(-(luma * lumaOffset) - (u * 128.0) - (v * 128.0));
        }

        return (weights, bias);
    }

    static TensorProto ShapedFloatTensor(string name, long[] dimensions, float[] values)
    {
        TensorProto tensor = FloatTensor(name, values);
        tensor.Dims.Clear();
        tensor.Dims.AddRange(dimensions);
        return tensor;
    }

    static void Add(GraphProto graph, string operation, string[] inputs, string[] outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { Name = $"{operation.ToLowerInvariant()}_{graph.Node.Count}", OpType = operation };
        node.Input.AddRange(inputs);
        node.Output.AddRange(outputs);
        node.Attribute.AddRange(attributes);
        graph.Node.Add(node);
    }
}
