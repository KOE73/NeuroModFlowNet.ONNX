using System.Buffers.Binary;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class PerspectiveGridBuilder
{
    public const string InputName = "image";
    public const string OutputName = "perspective";

    public static byte[] BuildU8Nhwc(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        ReadOnlySpan<float> targetToSourceMatrix)
    {
        var graph = CreateBaseGraph(
            sourceWidth,
            sourceHeight,
            outputWidth,
            outputHeight,
            channels,
            TensorProto.Types.DataType.Uint8,
            TensorProto.Types.DataType.Uint8,
            targetToSourceMatrix);

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

        return CreateModel("neuromodflownet-perspective-u8-nhwc-builder", graph).ToByteArray();
    }

    public static byte[] BuildNchw(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        TensorProto.Types.DataType elementType,
        ReadOnlySpan<float> targetToSourceMatrix)
    {
        if(elementType is not (TensorProto.Types.DataType.Float or TensorProto.Types.DataType.Float16))
            throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Perspective NCHW currently supports FP32 and FP16.");

        var graph = CreateBaseGraph(
            sourceWidth,
            sourceHeight,
            outputWidth,
            outputHeight,
            channels,
            elementType,
            elementType,
            targetToSourceMatrix);

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
            : "perspective_float";
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

        return CreateModel("neuromodflownet-perspective-nchw-builder", graph).ToByteArray();
    }

    static GraphProto CreateBaseGraph(
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        int channels,
        TensorProto.Types.DataType inputElementType,
        TensorProto.Types.DataType outputElementType,
        ReadOnlySpan<float> targetToSourceMatrix)
    {
        var graph = new GraphProto { Name = "perspective_grid_sample" };

        bool nhwc = inputElementType == TensorProto.Types.DataType.Uint8;
        graph.Input.Add(nhwc
            ? TensorInfo(InputName, inputElementType, 1, sourceHeight, sourceWidth, channels)
            : TensorInfo(InputName, inputElementType, 1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(nhwc
            ? TensorInfo(OutputName, outputElementType, 1, outputHeight, outputWidth, channels)
            : TensorInfo(OutputName, outputElementType, 1, channels, outputHeight, outputWidth));

        graph.Initializer.Add(GridTensor("grid", sourceWidth, sourceHeight, outputWidth, outputHeight, targetToSourceMatrix));
        return graph;
    }

    static void AddGridSample(GraphProto graph, string inputName, string outputName)
    {
        graph.Node.Add(new NodeProto
        {
            Name = "perspective_grid_sample",
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
        ReadOnlySpan<float> matrix)
    {
        if(matrix.Length < 9)
            throw new ArgumentException("Perspective matrix must contain 9 values.", nameof(matrix));

        var bytes = new byte[checked(outputHeight * outputWidth * 2 * sizeof(float))];
        int byteOffset = 0;

        for(int y = 0; y < outputHeight; y++)
        {
            for(int x = 0; x < outputWidth; x++)
            {
                float denominator = matrix[6] * x + matrix[7] * y + matrix[8];
                float sourceX = (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator;
                float sourceY = (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator;
                float normalizedX = NormalizeCoordinate(sourceX, sourceWidth);
                float normalizedY = NormalizeCoordinate(sourceY, sourceHeight);

                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(byteOffset, sizeof(float)), normalizedX);
                byteOffset += sizeof(float);
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(byteOffset, sizeof(float)), normalizedY);
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

    static float NormalizeCoordinate(float value, int size) =>
        size <= 1 ? 0f : 2f * value / (size - 1) - 1f;
}
