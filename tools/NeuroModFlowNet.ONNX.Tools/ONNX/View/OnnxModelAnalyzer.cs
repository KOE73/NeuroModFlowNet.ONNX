namespace NeuroModFlowNet.ONNX.Tools.ONNX;

public class OnnxModelAnalyzer_Console : OnnxModelToolBase
{


    public void InspectModel(string path)
    {
        ModelProto model = LoadModel(path);

        // Выводим информацию
        Console.WriteLine($"ONNX IR Version: {model.IrVersion}");
        Console.WriteLine($"ONNX IR Version: {model.IrVersion}");
        Console.WriteLine($"Producer: {model.ProducerName} v{model.ProducerVersion}");

        //PrintModelInputs(model);
        //Console.WriteLine("*******************************************************************");
        PrintFullGraphDetails(model);
        Console.WriteLine("*******************************************************************");



        //var t5 = model.Graph.Node.Take(5);
        //foreach(var n in t5)
        //    PrintNodeDetailsWithTypes(model, n);

        //Console.WriteLine($"First node: {firstNode.OpType} (Name: {firstNode.Name})");
    }

    public void PrintNodeDetailsWithTypes(ModelProto model, NodeProto node)
    {
        Console.WriteLine($"--- NODE: {node.Name} [{node.OpType}] ---");

        Console.WriteLine("Inputs:");
        foreach(var inputName in node.Input)
        {
            // 1. Ищем тип входа в Input модели (ValueInfoProto)
            var inputInfo = model.Graph.Input.FirstOrDefault(i => i.Name == inputName);
            string typeStr = "Unknown";

            if(inputInfo != null)
            {
                typeStr = GetDataTypeName(inputInfo.Type.TensorType.ElemType);
            }
            else
            {
                // 2. Если это не внешний вход, ищем в весах (Initializer)
                var initializer = model.Graph.Initializer.FirstOrDefault(i => i.Name == inputName);
                if(initializer != null)
                    typeStr = GetDataTypeName(initializer.DataType);
                else
                    // 3. Ищем в промежуточных значениях (ValueInfo)
                    typeStr = GetDataTypeName(model.Graph.ValueInfo.FirstOrDefault(v => v.Name == inputName)?.Type.TensorType.ElemType ?? 0);
            }

            Console.WriteLine($"  - {inputName} | Type: {typeStr}");
        }

        // Вывод атрибутов (остается прежним)
        Console.WriteLine("Attributes:");
        foreach(var attr in node.Attribute)
        {
            Console.WriteLine($"  * {attr.Name}: {GetAttributeValue(attr)}");
        }
    }




    private string GetAttributeValue(AttributeProto attr)
    {
        // attr.Type — это перечисление AttributeType
        return attr.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attr.F.ToString("F4"),
            AttributeProto.Types.AttributeType.Int => attr.I.ToString(),
            AttributeProto.Types.AttributeType.String => attr.S.ToStringUtf8(),
            AttributeProto.Types.AttributeType.Floats => $"[{string.Join(", ", attr.Floats)}]",
            AttributeProto.Types.AttributeType.Ints => $"[{string.Join(", ", attr.Ints)}]",
            AttributeProto.Types.AttributeType.Strings => $"[{string.Join(", ", attr.Strings.Select(s => s.ToStringUtf8()))}]",

            // Для тензоров и графов выводим только общую информацию
            AttributeProto.Types.AttributeType.Tensor => $"Tensor: {attr.T.Name} ({attr.T.DataType})",
            AttributeProto.Types.AttributeType.Graph => $"Graph: {attr.G.Name}",

            _ => $"Complex Type ({attr.Type})"
        };
    }


    public void PrintModelInputs(ModelProto model)
    {
        Console.WriteLine("=== MODEL INPUTS (SHAPES) ===");
        foreach(var input in model.Graph.Input)
        {
            // Иерархия: Input -> Type -> TensorType -> Shape -> Dim
            var dims = input.Type.TensorType.Shape.Dim;

            // Преобразуем коллекцию измерений в понятную строку
            var shapeParts = dims.Select(d =>
            {
                if(d.ValueCase == TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue) return d.DimValue.ToString(); // Фиксированный размер (например, 512)
                if(d.ValueCase == TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam) return d.DimParam;            // Динамический (например, "N" или "batch")
                return "?";                                                // Неизвестно
            });

            Console.WriteLine($"{input.Name}: [{string.Join(" x ", shapeParts)}]");
        }
    }

    public void PrintFullGraphDetails(ModelProto model)
    {
        var graph = model.Graph;

        // 1. Входы модели
        PrintValueInfoList("MODEL INPUTS", graph.Input);

        // 2. Выходы модели
        PrintValueInfoList("MODEL OUTPUTS", graph.Output);

        // 3. Промежуточные тензоры (ValueInfo)
        PrintValueInfoList("INTERMEDIATE TENSORS (VALUE_INFO)", graph.ValueInfo);

        // 4. Узлы и их связи
        Console.WriteLine("\n=== NODES AND FLOW ===");
        foreach(var node in graph.Node)
        {
            Console.WriteLine($"Node: {node.Name} [{node.OpType}]");

            foreach(var input in node.Input)
                Console.WriteLine($"  In:  {input} {GetShapeStr(model, input)}");

            foreach(var output in node.Output)
                Console.WriteLine($"  Out: {output} {GetShapeStr(model, output)}");
        }
    }

    private void PrintValueInfoList(string title, IEnumerable<ValueInfoProto> list)
    {
        Console.WriteLine($"\n=== {title} ===");
        foreach(var v in list)
        {
            var shape = v.Type.TensorType.Shape.Dim.Select(d => d.HasDimValue ? d.DimValue.ToString() : (d.DimParam ?? "?"));
            var type = GetDataTypeName(v.Type.TensorType.ElemType);
            Console.WriteLine($"{v.Name} | {type} | Shape: [{string.Join(" x ", shape)}]");
        }
    }

    private string GetShapeStr(ModelProto model, string tensorName)
    {
        // Объединяем все источники метаданных: входы, выходы и промежуточные тензоры
        var info = model.Graph.Input
            .Concat(model.Graph.Output)
            .Concat(model.Graph.ValueInfo)
            .FirstOrDefault(v => v.Name == tensorName);

        if(info == null) return " [Shape: N/A]";

        // 1. Извлекаем тип данных (индекс -> имя)
        var typeIndex = info.Type.TensorType.ElemType;
        string typeName = GetDataTypeName(typeIndex); // Используем твой метод с switch

        // 2. Извлекаем измерения (Shapes)
        var dims = info.Type.TensorType.Shape.Dim;
        string shapeStr = dims.Count == 0
            ? "Scalar"
            : string.Join("x", dims.Select(d => d.ValueCase switch
            {
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue => d.DimValue.ToString(),
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam => d.DimParam,
                _ => "?"
            }));

        return $" [{shapeStr}] {typeName}";
    }

    public void PrintNodeDetails(NodeProto node)
    {
        Console.WriteLine($"--- NODE: {node.Name} [{node.OpType}] ---");

        // Вывод входов
        Console.WriteLine("Inputs:");
        foreach(var input in node.Input)
            Console.WriteLine($"  - {input}");

        // Вывод выходов
        Console.WriteLine("Outputs:");
        foreach(var output in node.Output)
            Console.WriteLine($"  - {output}");

        // Вывод атрибутов
        Console.WriteLine("Attributes:");
        foreach(var attr in node.Attribute)
        {
            string val = attr.Type switch
            {
                AttributeProto.Types.AttributeType.Int => attr.I.ToString(),
                AttributeProto.Types.AttributeType.Ints => $"[{string.Join(", ", attr.Ints)}]",
                AttributeProto.Types.AttributeType.Float => attr.F.ToString(),
                AttributeProto.Types.AttributeType.String => attr.S.ToStringUtf8(),
                _ => $"Type: {attr.Type}"
            };
            Console.WriteLine($"  * {attr.Name}: {val}");
        }
    }


    #region helpers

    // Маппинг индексов ONNX в понятные названия
    private string GetDataTypeName(int typeIndex) => typeIndex switch
    {
        1 =>  "FLOAT  (Float32)",
        2 =>  "UINT8  (Byte)   ",
        3 =>  "INT8   (SByte)  ",
        10 => "FLOAT16(FP16)   ",
        11 => "DOUBLE (Float64)",
        _ => $"Undef  ({typeIndex,7})"
    };
    #endregion



}
