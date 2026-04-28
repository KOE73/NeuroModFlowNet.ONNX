namespace NeuroModFlowNet.ONNX.Tools.Modify;

[OnnxHead("ByteBGR_FP32", "Standard NHWC Byte BGR input head to FP32")]
public class OnnxModel_Injector_Preprocessing_Byte2FP32_BGR2RGB_Norm_0_1 : OnnxModelModifier
{
    public override string GetDefaultExtraName() => "_head32";

    // 1. ОПРЕДЕЛЯЕМ ИМЕНА ТЕНЗОРОВ
    const string New_RawByteBGR_InputName = "pixel_values";                             // U8 NHWC BGR
    const string Step1_CastToFP32_FP32_BGR_NHWC_OutputName = "pixels_fp32_bgr_nchw";    // FP32 NCHW RGB
    const string Step2_BGR2RGB_FP32_RGB_NHWC_OutputName = "pixels_fp32_nhwc";           // U8 NHWC RGB
    const string Step3_Transpose_FP32_RGB_NCHW_OutputName = "pixels_fp32_rgb_nhwc";     // U8 NCHW RGB

    const string ByteBGR_FP32_cast_fp32 = "ByteBGR_FP32_cast_fp32";
    const string ByteBGR_FP32_bgr_to_rgb = "ByteBGR_FP32_bgr_to_rgb";
    const string ByteBGR_FP32_transpose_nhwc_to_nchw = "ByteBGR_FP32_transpose_nhwc_to_nchw";
    const string ByteBGR_FP32_norm = "ByteBGR_FP32_norm";

    public override void Inject(ModelProto model)
    {
        var graph = model.Graph;
        var origInput = graph.Input[0]; // Оригинальный вход [1, 3, 512, 512] Это твой текущий "images"
        string origInputFirstConvName = origInput.Name;

        TensorShapeProto.Types.Dimension origDim0Batch = origInput.Type.TensorType.Shape.Dim[0];
        TensorShapeProto.Types.Dimension origDim1Channels = origInput.Type.TensorType.Shape.Dim[1];
        TensorShapeProto.Types.Dimension origDim2Height = origInput.Type.TensorType.Shape.Dim[2];
        TensorShapeProto.Types.Dimension origDim3Width = origInput.Type.TensorType.Shape.Dim[3];

        // 0. ВХОД: U8 NHWC [1, 512, 512, 3]
        var newRawByteBGRInput = new ValueInfoProto { Name = New_RawByteBGR_InputName };
        newRawByteBGRInput.Type = new TypeProto
        {
            TensorType = new TypeProto.Types.Tensor
            {
                ElemType = (int)TensorProto.Types.DataType.Uint8,
                Shape = new TensorShapeProto()
            }
        };

        // Копируем размеры, но ставим как в Mat [NHWC].
        // Только размеры, перестановки информации тут нет!
        newRawByteBGRInput.Type.TensorType.Shape.Dim.Add(origDim0Batch);      // N Batch
        newRawByteBGRInput.Type.TensorType.Shape.Dim.Add(origDim2Height);     // H
        newRawByteBGRInput.Type.TensorType.Shape.Dim.Add(origDim3Width);      // W
        newRawByteBGRInput.Type.TensorType.Shape.Dim.Add(origDim1Channels);   // Channels   (3)



        // 1. Node: CAST: U8 -> FP32
        var castNode = new NodeProto
        {
            Name = ByteBGR_FP32_cast_fp32,
            OpType = "Cast",
            Input = { New_RawByteBGR_InputName },
            Output = { Step1_CastToFP32_FP32_BGR_NHWC_OutputName },
            Attribute = { new AttributeProto { Name = "to", I = (long)TensorProto.Types.DataType.Float, Type = AttributeProto.Types.AttributeType.Int } }
        };


        // 2. GATHER (U8): BGR -> RGB по оси 3 (Перестановка каналов 2, 1, 0)
        // 2.1 Создаем индекс для Gather
        var gatherIndicesName = "indices_bgr2rgb";
        var gatherIniter = new TensorProto
        {
            Name = gatherIndicesName,
            DataType = (int)TensorProto.Types.DataType.Int64,
            Dims = { 3 },           // Это не ось! это размер масива индексов. У нас 3 канала, и мы хотим их переставить, поэтому 3.
            Int64Data = { 2, 1, 0 } // Меняем 0-й и 2-й каналы
        };
        graph.Initializer.Add(gatherIniter);

        // 2.2 Node 
        // Attribute I = 3: Указывает, что мы лезем именно в 3-ю ось(в каналы). Т.е. это размерность С в NHWC
        var gatherNode = new NodeProto
        {
            Name = ByteBGR_FP32_bgr_to_rgb,
            OpType = "Gather",
            Input = { Step1_CastToFP32_FP32_BGR_NHWC_OutputName, gatherIndicesName },
            Output = { Step2_BGR2RGB_FP32_RGB_NHWC_OutputName },
            Attribute = { new AttributeProto { Name = "axis", I = 3, Type = AttributeProto.Types.AttributeType.Int } }
        };


        // 3. TRANSPOSE (U8): NHWC -> NCHW [0, 3, 1, 2]
        var transNode = new NodeProto
        {
            Name = ByteBGR_FP32_transpose_nhwc_to_nchw,
            OpType = "Transpose",
            Input = { Step2_BGR2RGB_FP32_RGB_NHWC_OutputName },
            Output = { Step3_Transpose_FP32_RGB_NCHW_OutputName },
            Attribute = { new AttributeProto { Name = "perm", Ints = { 0, 3, 1, 2 }, Type = AttributeProto.Types.AttributeType.Ints } }
        };




        // 4. MUL (Нормализация 1/255)
        // 4.1 Создаем константу 0.00392157
        var normScaleName = "norm_scale_byte_to_0_1";
        var scaleIniter = new TensorProto
        {
            Name = normScaleName,
            DataType = (int)TensorProto.Types.DataType.Float,
            Dims = { 1 },
            RawData = Google.Protobuf.ByteString.CopyFrom(BitConverter.GetBytes(1f / 255f))
        };
        graph.Initializer.Add(scaleIniter);

        // 4.2 Node
        var mulNode = new NodeProto
        {
            Name = ByteBGR_FP32_norm,
            OpType = "Mul",
            Input = { Step3_Transpose_FP32_RGB_NCHW_OutputName, normScaleName },
            Output = { origInputFirstConvName }, // Подключаем к существующему входу Conv
        };


        // 7. ВАЖНО: Добавляем ValueInfo для промежуточных тензоров и самое главное их размеры
        graph.ValueInfo.Add(
        [
            // После Cast: [N, H, W, C]
            CreateValueInfo(Step1_CastToFP32_FP32_BGR_NHWC_OutputName, TensorProto.Types.DataType.Float, origDim0Batch, origDim2Height, origDim3Width, origDim1Channels),
            
            // После Gather: [N, H, W, C]
            CreateValueInfo(Step2_BGR2RGB_FP32_RGB_NHWC_OutputName, TensorProto.Types.DataType.Float, origDim0Batch, origDim2Height, origDim3Width, origDim1Channels),
            
            // После Transpose: [N, C, H, W]
            CreateValueInfo(Step3_Transpose_FP32_RGB_NCHW_OutputName, TensorProto.Types.DataType.Float, origDim0Batch, origDim1Channels, origDim2Height, origDim3Width),
            
            // Описываем "images", который теперь не вход, а результат Mul
            CreateValueInfo(origInputFirstConvName, TensorProto.Types.DataType.Float, origDim0Batch, origDim1Channels, origDim2Height, origDim3Width)
        ]);

        // 5. Обновляем граф
        // 5.1 Новый вход - это наш newRawByteBGRInput, а не origInput
        graph.Input.Clear();
        graph.Input.Add(newRawByteBGRInput);

        // 5.2 Вставляем новые узлы обработки в начало списка
        graph.Node.Insert(0, castNode);
        graph.Node.Insert(1, gatherNode);
        graph.Node.Insert(2, transNode);
        graph.Node.Insert(3, mulNode);
    }
    //    TensorProto.Types.DataType type,
    //    long[] shape)
    //{
    //    var vi = new ValueInfoProto { Name = name };
    //    var ts = new TypeProto.Types.Tensor { ElemType = (int)type };
    //    foreach(var dim in shape) ts.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dim });
    //    vi.Type = new TypeProto { TensorType = ts };
    //    return vi;
    //}
}
