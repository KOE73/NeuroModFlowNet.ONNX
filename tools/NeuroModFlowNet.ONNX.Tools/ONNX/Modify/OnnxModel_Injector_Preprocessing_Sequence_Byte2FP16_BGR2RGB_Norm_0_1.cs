namespace NeuroModFlowNet.ONNX.Tools.Modify;

[OnnxHead("SequenceByteBGR", "Sequence of Tensors (List of Mat) input head")]
public class OnnxModel_Injector_Preprocessing_Sequence_Byte2FP16_BGR2RGB_Norm_0_1 : OnnxModelModifier
{
    public override string GetDefaultExtraName() => "_head_seq";

    /// <summary>
    /// !!!! https://github.com/onnx/onnx-tensorrt/blob/10.14-GA/docs/operators.md
    /// </summary>
    /// <param name="model"></param>
    public override void Inject(ModelProto model)
    {
        var graph = model.Graph;
        var oldInput = graph.Input[0];
        string firstConvInputName = oldInput.Name;

        // 1. Создаем вход типа SEQUENCE(TENSOR)
        // Вместо одного тензора мы теперь принимаем список (последовательность)
        var seqInput = new ValueInfoProto { Name = "pixel_sequence" };
        seqInput.Type = new TypeProto
        {
            SequenceType = new TypeProto.Types.Sequence//TypeProto
            {
                ElemType = new TypeProto
                {
                    TensorType = new TypeProto.Types.Tensor
                    {
                        ElemType = (int)TensorProto.Types.DataType.Uint8,
                        Shape = new TensorShapeProto
                        {
                            // Каждая картинка в последовательности: [1, H, W, 3]
                            Dim = {
                                //new TensorShapeProto.Types.Dimension { DimValue = 1 },
                                //new TensorShapeProto.Types.Dimension { DimValue = 512 },
                                //new TensorShapeProto.Types.Dimension { DimValue = 512 },
                                //new TensorShapeProto.Types.Dimension { DimValue = 3 }
                                oldInput.Type.TensorType.Shape.Dim[0], // Batch
                                oldInput.Type.TensorType.Shape.Dim[1], // Canals
                                oldInput.Type.TensorType.Shape.Dim[2], // H
                                oldInput.Type.TensorType.Shape.Dim[3], // W
                            }
                        }
                    }
                }
            }
        };



        // 1. Описываем промежуточный тензор pixel_values (результат склейки)
        var pixelValuesValueInfo = new ValueInfoProto { Name = "pixel_values" };
        pixelValuesValueInfo.Type = new TypeProto
        {
            TensorType = new TypeProto.Types.Tensor
            {
                ElemType = 2, // UINT8
                Shape = new TensorShapeProto
                {
                    // Mat NHWC-> Yolo NCHW
                    Dim = {
                        oldInput.Type.TensorType.Shape.Dim[0], // Batch
                        oldInput.Type.TensorType.Shape.Dim[2], // H
                        oldInput.Type.TensorType.Shape.Dim[3], // W
                        oldInput.Type.TensorType.Shape.Dim[1], // Сhannels
                        //new TensorShapeProto.Types.Dimension { DimValue = 4 },
                        //new TensorShapeProto.Types.Dimension { DimValue = 512 },
                        //new TensorShapeProto.Types.Dimension { DimValue = 512 },
                        ////oldInput.Type.TensorType.Shape.Dim[2], // H (512)
                        ////oldInput.Type.TensorType.Shape.Dim[3], // W (512)
                        //new TensorShapeProto.Types.Dimension { DimValue = 3 }   // C (3)
                    }
                }
            }
        };


        graph.ValueInfo.Add(pixelValuesValueInfo);

        // 2. Узел ConcatFromSequence: "Склеиваем" список в один тензор [Batch, H, W, 3]
        var concatSeqNode = new NodeProto
        {
            OpType = "ConcatFromSequence",
            Input = { "pixel_sequence" },
            Output = { "pixel_values" },
            Name = "preprocess_concat_seq",
            Attribute = {
            new AttributeProto { Name = "axis", I = 0, Type = AttributeProto.Types.AttributeType.Int },
            new AttributeProto { Name = "new_axis", I = 0, Type = AttributeProto.Types.AttributeType.Int }
        }
        };

        // 3. Узел CAST: Byte -> FP16 (как у тебя)
        var castNode = new NodeProto
        {
            OpType = "Cast",
            Input = { "pixel_values" },
            Output = { "floated_pixels" },
            Name = "preprocess_cast",
            Attribute = { new AttributeProto { Name = "to", I = (long)TensorProto.Types.DataType.Float16, Type = AttributeProto.Types.AttributeType.Int } }
        };

        // 4. Узел TRANSPOSE: NHWC -> NCHW + BGR2RGB (если нужно)
        // Чтобы сделать BGR -> RGB, нужно использовать Gather, но пока оставим твой NHWC -> NCHW
        var transNode = new NodeProto
        {
            OpType = "Transpose",
            Input = { "floated_pixels" },
            Output = { "transposed_pixels" },
            Name = "preprocess_transpose",
            Attribute = { new AttributeProto { Name = "perm", Ints = { 0, 3, 1, 2 }, Type = AttributeProto.Types.AttributeType.Ints } }
        };

        // 5. Узел MUL (Нормализация 1/255)
        var scaleName = "norm_scale";
        var scaleIniter = new TensorProto
        {
            Name = scaleName,
            DataType = (int)TensorProto.Types.DataType.Float16,
            Dims = { 1 },
            // 1/255 в формате Half
            RawData = Google.Protobuf.ByteString.CopyFrom(BitConverter.GetBytes((Half)(1f / 255f)))
        };
        graph.Initializer.Add(scaleIniter);

        var mulNode = new NodeProto
        {
            OpType = "Mul",
            Input = { "transposed_pixels", scaleName },
            Output = { firstConvInputName },
            Name = "preprocess_norm"
        };

        // 6. Обновляем граф
        graph.Input.Clear();
        graph.Input.Add(seqInput); // Теперь вход — это последовательность

        // Вставляем узлы в правильном порядке
        graph.Node.Insert(0, concatSeqNode);
        graph.Node.Insert(1, castNode);
        graph.Node.Insert(2, transNode);
        graph.Node.Insert(3, mulNode);
    }

}
