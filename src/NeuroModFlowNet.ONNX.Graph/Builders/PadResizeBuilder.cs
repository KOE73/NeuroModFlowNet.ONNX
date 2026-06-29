using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

public static class PadResizeBuilder
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
        byte padValue)
    {
        var graph = new GraphProto { Name = "pad_resize_only" };

        graph.Input.Add(TensorInfo(
            InputName,
            TensorProto.Types.DataType.Uint8,
            1, sourceHeight, sourceWidth, channels));

        graph.Output.Add(TensorInfo(
            OutputName,
            TensorProto.Types.DataType.Uint8,
            1, outputHeight, outputWidth, channels));

        graph.Initializer.Add(Int64Tensor("sizes", [1, channels, resizedHeight, resizedWidth]));

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

        graph.Node.Add(new NodeProto
        {
            Name = "transpose_input_to_nchw",
            OpType = "Transpose",
            Input = { "image_float" },
            Output = { "image_nchw" },
            Attribute =
            {
                IntsAttribute("perm", [0, 3, 1, 2])
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "resize",
            OpType = "Resize",
            Input = { "image_nchw", "", "", "sizes" },
            Output = { "resized_nchw" },
            Attribute =
            {
                StringAttribute("mode", "linear"),
                StringAttribute("coordinate_transformation_mode", "asymmetric"),
                StringAttribute("nearest_mode", "round_prefer_floor")
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "transpose_output_to_nhwc",
            OpType = "Transpose",
            Input = { "resized_nchw" },
            Output = { "resized_float" },
            Attribute =
            {
                IntsAttribute("perm", [0, 2, 3, 1])
            }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "cast_resized_to_uint8",
            OpType = "Cast",
            Input = { "resized_float" },
            Output = { padLeft == 0 && padTop == 0 && padRight == 0 && padBottom == 0 ? OutputName : "resized_uint8" },
            Attribute =
            {
                IntAttribute("to", (long)TensorProto.Types.DataType.Uint8)
            }
        });

        AddPaddingNodes(
            graph,
            resizedWidth,
            resizedHeight,
            outputWidth,
            padLeft,
            padTop,
            padRight,
            padBottom,
            channels,
            padValue);

        return CreateModel("neuromodflownet-runtime-pad-resize-builder", graph).ToByteArray();
    }

    static void AddPaddingNodes(
        GraphProto graph,
        int resizedWidth,
        int resizedHeight,
        int outputWidth,
        int padLeft,
        int padTop,
        int padRight,
        int padBottom,
        int channels,
        byte padValue)
    {
        if(padLeft == 0 && padTop == 0 && padRight == 0 && padBottom == 0)
            return;

        string middleName = "resized_uint8";
        if(padLeft > 0 || padRight > 0)
        {
            var middleInputs = new List<string>();
            if(padLeft > 0)
            {
                graph.Initializer.Add(UInt8Tensor("left_pad", [1, resizedHeight, padLeft, channels], padValue));
                middleInputs.Add("left_pad");
            }

            middleInputs.Add("resized_uint8");

            if(padRight > 0)
            {
                graph.Initializer.Add(UInt8Tensor("right_pad", [1, resizedHeight, padRight, channels], padValue));
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
                    IntAttribute("axis", 2)
                }
            });
            middleName = "middle_with_width_padding";
        }

        var outputInputs = new List<string>();
        if(padTop > 0)
        {
            graph.Initializer.Add(UInt8Tensor("top_pad", [1, padTop, outputWidth, channels], padValue));
            outputInputs.Add("top_pad");
        }

        outputInputs.Add(middleName);

        if(padBottom > 0)
        {
            graph.Initializer.Add(UInt8Tensor("bottom_pad", [1, padBottom, outputWidth, channels], padValue));
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
                IntAttribute("axis", 1)
            }
        });
    }

    static TensorProto UInt8Tensor(string name, ReadOnlySpan<long> dimensions, byte value)
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Uint8
        };

        long itemCount = 1;
        foreach(long dimension in dimensions)
        {
            tensor.Dims.Add(dimension);
            itemCount = checked(itemCount * dimension);
        }

        tensor.RawData = ByteString.CopyFrom(Enumerable.Repeat(value, checked((int)itemCount)).ToArray());
        return tensor;
    }
}
