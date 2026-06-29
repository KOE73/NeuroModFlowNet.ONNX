using System.Buffers.Binary;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class PadResizeNchwBuilder
{
    public const string InputName = "image";
    public const string OutputName = "pad_resized";

    public static byte[] Build(
        int sourceWidth,
        int sourceHeight,
        int resizedWidth,
        int resizedHeight,
        int outputWidth,
        int outputHeight,
        int padLeft,
        int padTop,
        int padRight,
        int padBottom,
        int channels,
        TensorProto.Types.DataType elementType,
        float padValue)
    {
        if(elementType is not (TensorProto.Types.DataType.Float or TensorProto.Types.DataType.Float16))
            throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "NCHW pad-resize currently supports FP32 and FP16.");

        var graph = new GraphProto { Name = "pad_resize_nchw_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            elementType,
            1, channels, sourceHeight, sourceWidth));

        graph.Output.Add(TensorInfo(
            OutputName,
            elementType,
            1, channels, outputHeight, outputWidth));

        graph.Initializer.Add(Int64Tensor("sizes", [1, channels, resizedHeight, resizedWidth]));

        string resizeInputName = InputName;
        if(elementType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_input_to_float",
                OpType = "Cast",
                Input = { InputName },
                Output = { "image_float" },
                Attribute =
                {
                    IntAttribute("to", (long)TensorProto.Types.DataType.Float)
                }
            });
            resizeInputName = "image_float";
        }

        string resizeOutputName = elementType == TensorProto.Types.DataType.Float
            ? "resized_typed"
            : "resized_float";

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { resizeInputName, "", "", "sizes" },
            Output = { resizeOutputName },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "asymmetric"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        string paddingInputName = resizeOutputName;
        if(elementType == TensorProto.Types.DataType.Float16)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "cast_resized_to_float16",
                OpType = "Cast",
                Input = { resizeOutputName },
                Output = { "resized_typed" },
                Attribute =
                {
                    IntAttribute("to", (long)TensorProto.Types.DataType.Float16)
                }
            });
            paddingInputName = "resized_typed";
        }

        AddPaddingNodes(
            graph,
            paddingInputName,
            resizedWidth,
            resizedHeight,
            outputWidth,
            padLeft,
            padTop,
            padRight,
            padBottom,
            channels,
            elementType,
            padValue);

        return CreateModel("neuromodflownet-pad-resize-nchw-builder", graph).ToByteArray();
    }

    static void AddPaddingNodes(
        GraphProto graph,
        string inputName,
        int resizedWidth,
        int resizedHeight,
        int outputWidth,
        int padLeft,
        int padTop,
        int padRight,
        int padBottom,
        int channels,
        TensorProto.Types.DataType elementType,
        float padValue)
    {
        if(padLeft == 0 && padTop == 0 && padRight == 0 && padBottom == 0)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "identity_output",
                OpType = "Identity",
                Input = { inputName },
                Output = { OutputName }
            });
            return;
        }

        string middleName = inputName;
        if(padLeft > 0 || padRight > 0)
        {
            var middleInputs = new List<string>();
            if(padLeft > 0)
            {
                graph.Initializer.Add(TypedTensor("left_pad", [1, channels, resizedHeight, padLeft], elementType, padValue));
                middleInputs.Add("left_pad");
            }

            middleInputs.Add(inputName);

            if(padRight > 0)
            {
                graph.Initializer.Add(TypedTensor("right_pad", [1, channels, resizedHeight, padRight], elementType, padValue));
                middleInputs.Add("right_pad");
            }

            graph.Node.Add(new NodeProto
            {
                Name = "concat_width_padding",
                OpType = "Concat",
                Input = { middleInputs },
                Output = { "middle_with_width_padding" },
                Attribute =
                {
                    IntAttribute("axis", 3)
                }
            });
            middleName = "middle_with_width_padding";
        }

        var outputInputs = new List<string>();
        if(padTop > 0)
        {
            graph.Initializer.Add(TypedTensor("top_pad", [1, channels, padTop, outputWidth], elementType, padValue));
            outputInputs.Add("top_pad");
        }

        outputInputs.Add(middleName);

        if(padBottom > 0)
        {
            graph.Initializer.Add(TypedTensor("bottom_pad", [1, channels, padBottom, outputWidth], elementType, padValue));
            outputInputs.Add("bottom_pad");
        }

        if(outputInputs.Count == 1)
        {
            graph.Node.Add(new NodeProto
            {
                Name = "identity_output",
                OpType = "Identity",
                Input = { outputInputs[0] },
                Output = { OutputName }
            });
            return;
        }

        graph.Node.Add(new NodeProto
        {
            Name = "concat_height_padding",
            OpType = "Concat",
            Input = { outputInputs },
            Output = { OutputName },
            Attribute =
            {
                IntAttribute("axis", 2)
            }
        });
    }

    static TensorProto TypedTensor(
        string name,
        ReadOnlySpan<long> dimensions,
        TensorProto.Types.DataType elementType,
        float value)
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)elementType
        };

        long itemCount = 1;
        foreach(long dimension in dimensions)
        {
            tensor.Dims.Add(dimension);
            itemCount = checked(itemCount * dimension);
        }

        if(elementType == TensorProto.Types.DataType.Float)
        {
            var bytes = new byte[checked((int)itemCount * sizeof(float))];
            for(int index = 0; index < itemCount; index++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float)), value);

            tensor.RawData = ByteString.CopyFrom(bytes);
            return tensor;
        }

        if(elementType == TensorProto.Types.DataType.Float16)
        {
            var bytes = new byte[checked((int)itemCount * sizeof(ushort))];
            ushort halfBits = BitConverter.HalfToUInt16Bits((Half)value);
            for(int index = 0; index < itemCount; index++)
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * sizeof(ushort), sizeof(ushort)), halfBits);

            tensor.RawData = ByteString.CopyFrom(bytes);
            return tensor;
        }

        throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Unsupported pad tensor type.");
    }
}
