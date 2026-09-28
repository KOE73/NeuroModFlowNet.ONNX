using System.Buffers.Binary;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class UndistortGridBuilder
{
    public const string InputName = "image";
    public const string OutputName = "undistorted";

    public static byte[] BuildU8Nhwc(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        ReadOnlySpan<float> distortion)
    {
        var graph = CreateBaseGraph(
            sourceWidth,
            sourceHeight,
            outputWidth,
            outputHeight,
            channels,
            TensorProto.Types.DataType.Uint8,
            TensorProto.Types.DataType.Uint8,
            distortion);

        graph.Node.Add(new NodeProto
        {
            Name = "cast_input_to_float",
            OpType = "Cast",
            Input = { InputName },
            Output = { "image_float_nhwc" },
            Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Float) }
        });
        graph.Node.Add(new NodeProto
        {
            Name = "transpose_input_to_nchw",
            OpType = "Transpose",
            Input = { "image_float_nhwc" },
            Output = { "image_float_nchw" },
            Attribute = { IntsAttribute("perm", [0, 3, 1, 2]) }
        });
        AddGridSample(graph, "image_float_nchw", "sampled_float_nchw");
        graph.Node.Add(new NodeProto
        {
            Name = "transpose_output_to_nhwc",
            OpType = "Transpose",
            Input = { "sampled_float_nchw" },
            Output = { "sampled_float_nhwc" },
            Attribute = { IntsAttribute("perm", [0, 2, 3, 1]) }
        });
        graph.Node.Add(new NodeProto
        {
            Name = "cast_output_to_uint8",
            OpType = "Cast",
            Input = { "sampled_float_nhwc" },
            Output = { OutputName },
            Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Uint8) }
        });

        return CreateModel("neuromodflownet-undistort-u8-nhwc-builder", graph).ToByteArray();
    }

    public static byte[] BuildNchw(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        TensorProto.Types.DataType elementType,
        ReadOnlySpan<float> distortion)
    {
        if(elementType is not (TensorProto.Types.DataType.Float or TensorProto.Types.DataType.Float16))
            throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Undistort NCHW currently supports FP32 and FP16.");

        var graph = CreateBaseGraph(
            sourceWidth,
            sourceHeight,
            outputWidth,
            outputHeight,
            channels,
            elementType,
            elementType,
            distortion);

        string gridInputName = InputName;
        if(elementType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_input_to_float",
                OpType = "Cast",
                Input = { InputName },
                Output = { "image_float" },
                Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Float) }
            });
            gridInputName = "image_float";
        }

        string gridOutputName = elementType == TensorProto.Types.DataType.Float
            ? OutputName
            : "undistorted_float";
        AddGridSample(graph, gridInputName, gridOutputName);

        if(elementType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_output_to_float16",
                OpType = "Cast",
                Input = { gridOutputName },
                Output = { OutputName },
                Attribute = { IntAttribute("to", (long)TensorProto.Types.DataType.Float16) }
            });
        }

        return CreateModel("neuromodflownet-undistort-nchw-builder", graph).ToByteArray();
    }

    static GraphProto CreateBaseGraph(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        TensorProto.Types.DataType inputElementType,
        TensorProto.Types.DataType outputElementType,
        ReadOnlySpan<float> distortion)
    {
        var graph = new GraphProto { Name = "undistort_grid_sample" };

        bool nhwc = inputElementType == TensorProto.Types.DataType.Uint8;
        graph.Input.Add(nhwc
            ? TensorInfo(InputName, inputElementType, 1, sourceHeight, sourceWidth, channels)
            : TensorInfo(InputName, inputElementType, 1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(nhwc
            ? TensorInfo(OutputName, outputElementType, 1, outputHeight, outputWidth, channels)
            : TensorInfo(OutputName, outputElementType, 1, channels, outputHeight, outputWidth));

        graph.Initializer.Add(GridTensor("grid", sourceWidth, sourceHeight, outputWidth, outputHeight, distortion));
        return graph;
    }

    static void AddGridSample(GraphProto graph, string inputName, string outputName)
    {
        graph.Node.Add(new NodeProto
        {
            Name = "undistort_grid_sample",
            OpType = "GridSample",
            Input = { inputName, "grid" },
            Output = { outputName },
            Attribute =
            {
                IntAttribute("align_corners", 1),
                StringAttribute("mode", "bilinear"),
                StringAttribute("padding_mode", "zeros")
            }
        });
    }

    static TensorProto GridTensor(
        string name,
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        ReadOnlySpan<float> distortion)
    {
        if(distortion.Length is not (9 or 13))
            throw new ArgumentException("Undistort distortion parameters must contain 9 values (source fx, fy, cx, cy, k1, k2, p1, p2, k3) or 13 values (plus output fx, fy, cx, cy).", nameof(distortion));

        var bytes = new byte[checked(outputHeight * outputWidth * 2 * sizeof(float))];
        int byteOffset = 0;

        for(int y = 0; y < outputHeight; y++)
        {
            for(int x = 0; x < outputWidth; x++)
            {
                WriteSourceCoordinate(bytes.AsSpan(byteOffset, sizeof(float)), NormalizeCoordinate(GetDistortedX(x, y, distortion), sourceWidth));
                byteOffset += sizeof(float);
                WriteSourceCoordinate(bytes.AsSpan(byteOffset, sizeof(float)), NormalizeCoordinate(GetDistortedY(x, y, distortion), sourceHeight));
                byteOffset += sizeof(float);
            }
        }

        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
            RawData = ByteString.CopyFrom(bytes)
        };
        tensor.Dims.Add(1);
        tensor.Dims.Add(outputHeight);
        tensor.Dims.Add(outputWidth);
        tensor.Dims.Add(2);
        return tensor;
    }

    // Layout: [0..3] source fx, fy, cx, cy; [4..8] k1, k2, p1, p2, k3; optional [9..12] output fx, fy, cx, cy.
    // Output pixels are normalized with the output camera, distorted, then projected with the source camera.
    static float OutputFx(ReadOnlySpan<float> distortion) => distortion.Length >= 13 ? distortion[9] : distortion[0];

    static float OutputFy(ReadOnlySpan<float> distortion) => distortion.Length >= 13 ? distortion[10] : distortion[1];

    static float OutputCx(ReadOnlySpan<float> distortion) => distortion.Length >= 13 ? distortion[11] : distortion[2];

    static float OutputCy(ReadOnlySpan<float> distortion) => distortion.Length >= 13 ? distortion[12] : distortion[3];

    static float GetDistortedX(int x, int y, ReadOnlySpan<float> distortion)
    {
        float normalizedX = (x - OutputCx(distortion)) / OutputFx(distortion);
        float normalizedY = (y - OutputCy(distortion)) / OutputFy(distortion);
        float radius2 = normalizedX * normalizedX + normalizedY * normalizedY;
        float radius4 = radius2 * radius2;
        float radius6 = radius4 * radius2;
        float radial = 1f + distortion[4] * radius2 + distortion[5] * radius4 + distortion[8] * radius6;
        float distortedX = normalizedX * radial +
            2f * distortion[6] * normalizedX * normalizedY +
            distortion[7] * (radius2 + 2f * normalizedX * normalizedX);
        return distortedX * distortion[0] + distortion[2];
    }

    static float GetDistortedY(int x, int y, ReadOnlySpan<float> distortion)
    {
        float normalizedX = (x - OutputCx(distortion)) / OutputFx(distortion);
        float normalizedY = (y - OutputCy(distortion)) / OutputFy(distortion);
        float radius2 = normalizedX * normalizedX + normalizedY * normalizedY;
        float radius4 = radius2 * radius2;
        float radius6 = radius4 * radius2;
        float radial = 1f + distortion[4] * radius2 + distortion[5] * radius4 + distortion[8] * radius6;
        float distortedY = normalizedY * radial +
            distortion[6] * (radius2 + 2f * normalizedY * normalizedY) +
            2f * distortion[7] * normalizedX * normalizedY;
        return distortedY * distortion[1] + distortion[3];
    }

    static void WriteSourceCoordinate(Span<byte> destination, float value) =>
        BinaryPrimitives.WriteSingleLittleEndian(destination, value);

    static float NormalizeCoordinate(float value, int size) =>
        size <= 1 ? 0f : 2f * value / (size - 1) - 1f;
}
