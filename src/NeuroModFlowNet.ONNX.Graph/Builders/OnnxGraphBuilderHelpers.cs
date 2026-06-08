namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// EN: Helper methods for programmatically constructing ONNX model structures (Protobuf representations) at runtime.
/// RU: Вспомогательные методы для программного конструирования структур ONNX-модели (Protobuf-представлений) в рантайме.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Simplifies the generation of ONNX model components (tensors, attributes, and graph metadata) directly from C# code.
/// Reasons: Programmatic construction allows generating lightweight preprocessing sub-graphs (like resizing or cropping) on the fly, 
/// avoiding static model storage, external dependencies, or complex manual protobuf nesting.
/// 
/// RU:
/// Суть: Упрощает генерацию элементов ONNX-модели (тензоров, атрибутов и метаданных графа) напрямую из C#-кода.
/// Причины: Динамическое построение позволяет «на лету» формировать легковесные подграфы предобработки (например, изменение размера или кроп),
/// исключая необходимость хранения статических файлов моделей, внешних зависимостей или написания сложного вложенного protobuf-кода вручную.
/// </remarks>
public static class OnnxGraphBuilderHelpers
{
    /// <summary>
    /// EN: Creates a ValueInfoProto describing the metadata of a tensor.
    /// RU: Создает ValueInfoProto, описывающий метаданные тензора.
    /// </summary>
    /// <remarks>
    /// EN:
    /// Essence: Constructs type and shape declarations for ONNX graph inputs or outputs.
    /// Reasons: Encapsulates the verbose nested Protobuf structure (TypeProto -> TensorType -> Shape -> Dim) required by the ONNX specification.
    /// 
    /// RU:
    /// Суть: Формирует описание типа и формы для входов и выходов ONNX-графа.
    /// Причины: Инкапсулирует многословную вложенную структуру Protobuf (TypeProto -> TensorType -> Shape -> Dim), требуемую спецификацией ONNX.
    /// </remarks>
    public static ValueInfoProto TensorInfo(
        string name,
        TensorProto.Types.DataType type,
        params long[] dimensions)
    {
        var shape = new TensorShapeProto();

        foreach(long dimension in dimensions)
            shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dimension });

        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor
                {
                    ElemType = (int)type,
                    Shape = shape
                }
            }
        };
    }

    /// <summary>
    /// EN: Creates a TensorProto containing 64-bit integer values.
    /// RU: Создает TensorProto, содержащий 64-битные целые числа.
    /// </summary>
    /// <remarks>
    /// EN:
    /// Essence: Generates a constant tensor (initializer) filled with Int64 values.
    /// Reasons: Many ONNX operations (like Slice indices, axes, or strides) accept inputs/parameters as tensors rather than node attributes. 
    /// Using ReadOnlySpan eliminates array allocations on the heap.
    /// 
    /// RU:
    /// Суть: Формирует константный тензор (инициализатор), заполненный значениями Int64.
    /// Причины: Многие операции ONNX (например, индексы, оси или шаги Slice) принимают параметры в виде тензоров, а не атрибутов узла. 
    /// Использование ReadOnlySpan исключает аллокации массивов в куче.
    /// </remarks>
    public static TensorProto Int64Tensor(string name, ReadOnlySpan<long> values)
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Int64
        };

        tensor.Dims.Add(values.Length);
        foreach(long value in values)
            tensor.Int64Data.Add(value);

        return tensor;
    }

    /// <summary>
    /// EN: Creates a TensorProto containing 32-bit floating-point values.
    /// RU: Создает TensorProto, содержащий 32-битные числа с плавающей точкой.
    /// </summary>
    /// <remarks>
    /// EN:
    /// Essence: Generates a constant tensor (initializer) filled with Float values.
    /// Reasons: Required for numerical constants inside the graph, such as coordinates or scaling/normalization factors. 
    /// Using ReadOnlySpan avoids heap allocations.
    /// 
    /// RU:
    /// Суть: Формирует константный тензор (инициализатор), заполненный значениями Float.
    /// Причины: Необходим для числовых констант внутри графа, таких как координаты или масштабные коэффициенты нормализации. 
    /// Использование ReadOnlySpan предотвращает аллокации в куче.
    /// </remarks>
    public static TensorProto FloatTensor(string name, ReadOnlySpan<float> values)
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float
        };

        tensor.Dims.Add(values.Length);
        foreach(float value in values)
            tensor.FloatData.Add(value);

        return tensor;
    }

    /// <summary>
    /// EN: Creates an AttributeProto with a string value.
    /// RU: Создает AttributeProto со строковым значением.
    /// </summary>
    /// <remarks>
    /// EN:
    /// Essence: Constructs a string attribute configuration for an ONNX node or model.
    /// Reasons: Used to supply parameters that the ONNX specification defines as string attributes (e.g., interpolation modes for Resize).
    /// 
    /// RU:
    /// Суть: Формирует строковый атрибут конфигурации для узла или модели ONNX.
    /// Причины: Применяется для передачи параметров, которые по спецификации ONNX задаются строками (например, режимы интерполяции для Resize).
    /// </remarks>
    public static AttributeProto StringAttribute(string name, string value)
    {
        return new AttributeProto
        {
            Name = name,
            Type = AttributeProto.Types.AttributeType.String,
            S = ByteString.CopyFromUtf8(value)
        };
    }

    /// <summary>
    /// EN: Creates an AttributeProto with one integer value.
    /// RU: Создает AttributeProto с одним целочисленным значением.
    /// </summary>
    /// <remarks>
    /// EN: Used by operators such as Cast and Gather where ONNX stores small control values as attributes.
    /// RU: Применяется в операторах вроде Cast и Gather, где ONNX хранит небольшие управляющие значения в атрибутах.
    /// </remarks>
    public static AttributeProto IntAttribute(string name, long value)
    {
        return new AttributeProto
        {
            Name = name,
            Type = AttributeProto.Types.AttributeType.Int,
            I = value
        };
    }

    /// <summary>
    /// EN: Creates an AttributeProto with a list of integer values.
    /// RU: Создает AttributeProto со списком целочисленных значений.
    /// </summary>
    /// <remarks>
    /// EN: Used for attributes such as Transpose perm. Keeping it here avoids repeating protobuf boilerplate in every
    /// generated graph builder.
    /// RU: Используется для атрибутов вроде Transpose perm. Вынос сюда убирает повторение protobuf-шаблона в каждом
    /// построителе генерируемого графа.
    /// </remarks>
    public static AttributeProto IntsAttribute(string name, ReadOnlySpan<long> values)
    {
        var attribute = new AttributeProto
        {
            Name = name,
            Type = AttributeProto.Types.AttributeType.Ints
        };

        foreach(long value in values)
            attribute.Ints.Add(value);

        return attribute;
    }

    /// <summary>
    /// EN: Wraps a GraphProto into a fully conformant ONNX model.
    /// RU: Упаковывает GraphProto в валидный контейнер ONNX-модели.
    /// </summary>
    /// <remarks>
    /// EN:
    /// Essence: Creates a ModelProto container with specified metadata and imported operator sets (opsets).
    /// Reasons: A raw graph is not executable by ONNX runtimes; it must be wrapped in a model that explicitly declares the IR version 
    /// and operator set versions to guarantee execution compatibility.
    /// 
    /// RU:
    /// Суть: Создает контейнер ModelProto с указанием метаданных и импортируемых наборов операторов (opset).
    /// Причины: Сырой граф не может быть запущен средой выполнения ONNX; его необходимо обернуть в модель, явно заявляющую версии IR 
    /// и наборов операторов для обеспечения совместимости при запуске.
    /// </remarks>
    public static ModelProto CreateModel(string producerName, GraphProto graph, long opsetVersion = 18)
    {
        var model = new ModelProto
        {
            IrVersion = 9,
            ProducerName = producerName,
            Graph = graph
        };

        model.OpsetImport.Add(new OperatorSetIdProto
        {
            Domain = "",
            Version = opsetVersion
        });

        return model;
    }
}
