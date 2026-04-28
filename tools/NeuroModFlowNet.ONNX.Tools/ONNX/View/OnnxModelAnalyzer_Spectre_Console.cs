using Spectre.Console;
using Spectre.Console.Rendering;

namespace NeuroModFlowNet.ONNX.Tools.ONNX;

public class OnnxModelAnalyzer_Spectre_Console : OnnxModelToolBase
{
    public OnnxModelAnalyzer_Spectre_Console(
        HashSet<char>? iovt = null,
        int topN = -1,
        int bottomN = -1)
    {
        IOVT = iovt ?? ['I', 'O', 'V', 'T', 'A'];
        TopN = topN;
        BottomN = bottomN;
    }

    public HashSet<char> IOVT { get; set; }
    public int TopN { get; set; }
    public int BottomN { get; set; }

    public bool ShowBaseInfo { get; set; } = true;
    public bool ShowMetadataInfo { get; set; } = true;
    public bool ShowOpSetInfo { get; set; } = true;
    public bool ShowInputOutputInfo { get; set; } = true;

    public bool ShowNodeIONames { get; set; } = false;
    public bool ShowSplits { get; set; } = true;

    public void InspectModel(string path)
    {
        InitColors();
        ModelProto model = LoadModel(path);


        var rootTable = new Table().Border(TableBorder.None).HideHeaders();
        rootTable.AddColumn("Property");
        rootTable.AddColumn("Value");

        if(ShowBaseInfo) rootTable.AddRow(new Markup("[yellow]Info[/]"), TopInfoAsGrid(model));
        if(ShowMetadataInfo) rootTable.AddRow(new Markup("[yellow]Custom Metadata[/]"), MetadataAsGrid(model));
        if(ShowOpSetInfo) rootTable.AddRow(new Markup("[yellow]Opset Imports[/]"), OpsetAsTable(model));
        if(ShowInputOutputInfo) rootTable.AddRow(new Markup("[yellow]I/O Signature[/]"), InputOutputInfoAsTable(model, false));

        // Финальный вывод в красивой панели
        AnsiConsole.Write(
            new Panel(rootTable)
                .Header($"[bold white]ONNX Model: {model.Graph.Name}[/]")
                .Expand()
                .Border(BoxBorder.Double)
                .BorderStyle(new Style(Color.Green))
        );


        PrintFullGraphDetails(model);

    }


    public void PrintNodeDetailsWithTypes(ModelProto model, NodeProto node)
    {
        AnsiConsole.WriteLine($"--- NODE: {node.Name} [{node.OpType}] ---");

        AnsiConsole.WriteLine("Inputs:");
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

            AnsiConsole.WriteLine($"  - {inputName} | Type: {typeStr}");
        }

        // Вывод атрибутов (остается прежним)
        AnsiConsole.WriteLine("Attributes:");
        foreach(var attr in node.Attribute)
        {
            AnsiConsole.WriteLine($"  * {attr.Name}: {GetAttributeValue(attr.Name, attr)}");
        }
    }



    public void PrintFullGraphDetails(ModelProto model)
    {
        RepeatedField<NodeProto> nodes = model.Graph.Node;
        int nodeCount = nodes.Count;

        if(TopN > nodeCount) TopN = nodeCount;
        if(BottomN > nodeCount) BottomN = nodeCount;

        int maxNodeNameLength = 0;
        if(TopN > 0) maxNodeNameLength = Math.Max(maxNodeNameLength, nodes.Take(TopN).Max(o => o.Name.Length));
        if(BottomN > 0) maxNodeNameLength = Math.Max(maxNodeNameLength, nodes.Skip(Math.Max(0, nodes.Count - BottomN)).Max(o => o.Name.Length)); ;
        if(maxNodeNameLength == 0) maxNodeNameLength = nodes.Max(o => o.Name.Length);

        int consoleWidth = AnsiConsole.Console.Profile.Width;
        int colNodeNameWidth = maxNodeNameLength; // maxNodeNameLength * 2 / 3

        var nodeTable = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Blue)
            .Title("[bold blue]GRAPH NODES AND DATA FLOW[/]")
            .AddColumn("")
            .AddColumn(new TableColumn("[grey]Node Name[/]").Width(colNodeNameWidth)/*.NoWrap()*/)
            //.AddColumn(new TableColumn("[grey]Node Name[/]")/*.NoWrap()*/)
            .AddColumn(new TableColumn("[grey]Node Type[/]").NoWrap())
            .AddColumn(new TableColumn("[bold green]Inputs[/]"))
            .AddColumn(new TableColumn("[bold yellow]Outputs[/]"));

        int n = 0;
        bool emtyRowInserted = false;
        foreach(var node in nodes)
        {
            HashSet<char> factIOVT = [];
            // Создаем сетку для входов (без рамок)
            var inputGrid = new Grid().AddColumn().AddColumn().AddColumn();
            if(node.Input.Count > 0)
            {
                foreach(var input in node.Input)
                    factIOVT.UnionWith(GetShapeStr(model, input, inputGrid));
            }
            else
                inputGrid.AddRow(new Markup("[grey]none[/]"));

            if(node.Attribute.Count > 0 && IOVT.Contains('A'))
            {
                factIOVT.Add('A');
                foreach(var attr in node.Attribute)
                {
                    inputGrid.AddRow(
                        new Text("A", styleAttrColor),
                        new Markup($"[grey]{attr.Name}[/]", styleAttrColor),
                        new Markup($"[bold yellow]= {GetAttributeValue(attr.Name, attr)}[/]", styleDimGray)
                    );
                }
            }

            // Создаем сетку для выходов
            var outputGrid = new Grid().AddColumn().AddColumn().AddColumn();
            if(node.Output.Count > 0)
            {
                foreach(var output in node.Output)
                    factIOVT.UnionWith(GetShapeStr(model, output, outputGrid));
            }
            else
                outputGrid.AddRow(new Markup("[grey]none[/]"));

            bool showTop = TopN == -1 || TopN >= n;
            bool showBottom = BottomN == -1 || BottomN > (nodeCount - n);

            if((showTop || showBottom) && IOVT.Overlaps(factIOVT))
            {
                nodeTable.AddRow(
                    new Markup($"{n,3}"),
                    new Markup($"[white]{node.Name}[/]"),
                    new Markup($"[cyan]({node.OpType})[/]"),
                    inputGrid,
                    outputGrid
                );
                emtyRowInserted = false;
            }
            else
            {
                if(!emtyRowInserted)
                {
                    if(ShowSplits)
                        nodeTable.AddRow("~~~");//nodeTable.AddEmptyRow();
                    emtyRowInserted = true;
                }
            }

            n++;
        }


        AnsiConsole.Write(nodeTable);
    }

    HashSet<char> GetShapeStr(ModelProto model, string ioName, Grid grid)
    {
        HashSet<char> chars = [];

        Process('I', styleInColor, model.Graph.Input);
        Process('O', styleOutColor, model.Graph.Output);
        Process('V', styleValColor, model.Graph.ValueInfo);
        ProcessTensor('T', styleTensorColor, model.Graph.Initializer);

        return chars;

        void Process(char type, Style style, RepeatedField<ValueInfoProto> list)
        {
            if(IOVT.Contains(type)) AddRow(type, style, list.FirstOrDefault(v => v.Name == ioName));
        }
        void ProcessTensor(char type, Style style, RepeatedField<TensorProto> list)
        {
            if(IOVT.Contains(type)) AddRowTensor(type, style, list.FirstOrDefault(v => v.Name == ioName));
        }

        void AddRow(char type, Style style, ValueInfoProto? valueInfo)
        {
            if(valueInfo is null) return;
            chars.Add(type);
            (string typeStr, string shapeStr) = ValueInfoProtoToStr(valueInfo);


            grid.AddRow(
                new Text(ShowNodeIONames ? ioName : type.ToString(), style),
                new Markup(typeStr, style),
                new Markup($"{shapeStr}", styleDimGray)
                );

            //grid.AddRow(
            //    new Text(ShowNodeIONames ? ioName : type.ToString(), style),
            //    new Markup(GetDataTypeName(valueInfo.Type.TensorType.ElemType), style),
            //    new Markup($"[[{GetDimsStr(valueInfo.Type.TensorType.Shape.Dim)}]]", styleDimGray)
            //    );
        }
        void AddRowTensor(char type, Style style, TensorProto? info)
        {
            if(info is null) return;
            chars.Add(type);
            grid.AddRow(
                new Text(ShowNodeIONames ? ioName : type.ToString(), style),
                new Markup(GetDataTypeName(info.DataType), style),
                new Markup($"[[{GetTensorDimsStr(info.Dims)}]] {GetTensorValue(info)}", styleDimGray)
                );
        }



    }





    private string GetAttributeValue(string name, AttributeProto attr)
    {
        // attr.Type — это перечисление AttributeType
        return attr.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attr.F.ToString("F4"),
            AttributeProto.Types.AttributeType.Int => IntAttrStr(attr),
            AttributeProto.Types.AttributeType.String => attr.S.ToStringUtf8(),
            AttributeProto.Types.AttributeType.Floats => $"[[{string.Join(", ", attr.Floats)}]]",
            AttributeProto.Types.AttributeType.Ints => $"[[{string.Join(", ", attr.Ints)}]]",
            AttributeProto.Types.AttributeType.Strings => $"[[{string.Join(", ", attr.Strings.Select(s => s.ToStringUtf8()))}]]",

            // Для тензоров и графов выводим только общую информацию
            AttributeProto.Types.AttributeType.Tensor => $"Tensor: {attr.T.Name} ({attr.T.DataType})",
            AttributeProto.Types.AttributeType.Graph => $"Graph: {attr.G.Name}",

            _ => $"Complex Type ({attr.Type})"
        };

        string IntAttrStr(AttributeProto attr)
        {
            if(name == "to")
                return GetDataTypeName((int)attr.I);
            return attr.I.ToString();
        }
    }




    string GetDimsStr(Google.Protobuf.Collections.RepeatedField<TensorShapeProto.Types.Dimension> dims)
    {
        return dims.Count == 0
            ? "Scalar"
            : string.Join("x", dims.Select(d => d.ValueCase switch
            {
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue => $"[{dimValueColor}]{d.DimValue}[/]",
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam => $"[{dimParamColor}]{d.DimParam}[/]",
                _ => "?"
            }));
    }

    // НОВАЯ перегрузка для весов (TensorProto / Initializer)
    string GetTensorDimsStr(IEnumerable<long> dims)
    {
        // У весов всегда DimValue, поэтому используем dimValueColor
        return !dims.Any()
            ? "Scalar"
            : string.Join("x", dims.Select(d => $"[{dimValueColor}]{d}[/]"));
    }


    private string GetTensorValue(TensorProto tensor)
    {
        // Считаем общее количество элементов
        long count = tensor.Dims.Aggregate(1L, (a, b) => a * b);

        // Если тензор слишком большой (например, веса свертки), не выводим содержимое
        if(count > 10 || count == 0) return string.Empty;

        // Извлекаем данные в зависимости от типа
        List<string> values = new();

        // ONNX хранит данные либо в типизированных полях, либо в RawData
        if(tensor.Int64Data.Count > 0) values.AddRange(tensor.Int64Data.Select(x => x.ToString()));
        else if(tensor.Int32Data.Count > 0) values.AddRange(tensor.Int32Data.Select(x => x.ToString()));
        else if(tensor.FloatData.Count > 0) values.AddRange(tensor.FloatData.Select(x => x.ToString("F4")));
        else if(!tensor.RawData.IsEmpty)
        {
            // Если данные в RawData, нужно их распарсить (упрощенный пример для I64/I32)
            var span = tensor.RawData.Span;
            if(tensor.DataType == (int)TensorProto.Types.DataType.Int64)
                for(int i = 0; i < count; i++) values.Add(BitConverter.ToInt64(span.Slice(i * 8, 8)).ToString());
            else if(tensor.DataType == (int)TensorProto.Types.DataType.Int32)
                for(int i = 0; i < count; i++) values.Add(BitConverter.ToInt32(span.Slice(i * 4, 4)).ToString());
            else if(tensor.DataType == (int)TensorProto.Types.DataType.Float)
                for(int i = 0; i < count; i++) values.Add(BitConverter.ToSingle(span.Slice(i * 4, 4)).ToString("F4"));
        }

        if(values.Count == 0) return string.Empty;

        // Возвращаем в формате "= {значение}"
        return $" [bold yellow]= {string.Join(", ", values)}[/]";
    }


    public void PrintNodeDetails(NodeProto node)
    {
        AnsiConsole.WriteLine($"--- NODE: {node.Name} [{node.OpType}] ---");

        // Вывод входов
        AnsiConsole.WriteLine("Inputs:");
        foreach(var input in node.Input)
            AnsiConsole.WriteLine($"  - {input}");

        // Вывод выходов
        AnsiConsole.WriteLine("Outputs:");
        foreach(var output in node.Output)
            AnsiConsole.WriteLine($"  - {output}");

        // Вывод атрибутов
        AnsiConsole.WriteLine("Attributes:");
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
            AnsiConsole.WriteLine($"  * {attr.Name}: {val}");
        }
    }


    #region Colors

    string inColor = "green";
    string outColor = "yellow";
    string valColor = "blue";
    string tensorColor = "orange1";
    string attrColor = "magenta1";
    string dimValueColor = "Green3";
    string dimParamColor = "orange1";
    string dimGray = "Grey35";

    Style styleInColor = default!;
    Style styleOutColor = default!;
    Style styleValColor = default!;
    Style styleTensorColor = default!;
    Style styleAttrColor = default!;
    Style styleDimValueColor = default!;
    Style styleDimParamColor = default!;
    Style styleDimGray = default!;

    void InitColors()
    {
        styleInColor = Style.Parse(inColor);
        styleOutColor = Style.Parse(outColor);
        styleValColor = Style.Parse(valColor);
        styleTensorColor = Style.Parse(tensorColor);
        styleAttrColor = Style.Parse(attrColor);
        styleDimValueColor = Style.Parse(dimValueColor);
        styleDimParamColor = Style.Parse(dimParamColor);
        styleDimGray = Style.Parse(dimGray);
    }

    #endregion

    #region * As x

    static IRenderable TopInfoAsGrid(ModelProto model) =>
       new Grid()
           .AddColumn()
           .AddColumn()
           .AddRow("[DarkOliveGreen1_1]IR Version:[/]", model.IrVersion.ToString())
           .AddRow("[DarkOliveGreen1_1]Producer:[/]", $"[green]{model.ProducerName}[/] [yellow]v{model.ProducerVersion}[/]")
           .AddRow("[DarkOliveGreen1_1]Domain:[/]", model.Domain ?? "ai.onnx")
           .AddRow("[DarkOliveGreen1_1]Model Version:[/]", model.ModelVersion.ToString())
           .AddRow("[DarkOliveGreen1_1]Description:[/]", model.DocString ?? "[DarkSeaGreen2_1]N/A[/]");

    static IRenderable MetadataAsGrid(ModelProto model)
    {
        var metaTable = new Grid().AddColumn().AddColumn();

        foreach(var prop in model.MetadataProps)
            metaTable.AddRow(new Text(prop.Key, Style.Parse("Gold1")), new Text(prop.Value));

        return metaTable;
    }

    static Table OpsetAsTable(ModelProto model)
    {
        var opsetTable = new Table()
            .AddColumn("Domain")
            .AddColumn("Version")
            .Border(TableBorder.Minimal)
            //.Expand()
            //.PadTop(0)
            //.PadBottom(0)
            ;

        foreach(var import in model.OpsetImport)
        {
            string domainName = import.Domain switch
            {
                "" or null => "[bold cyan]ai.onnx (Standard)[/]",
                "ai.onnx.ml" => "[yellow]ai.onnx.ml (Classic ML)[/]",
                var d => $"[purple]{d}[/]" // Кастомный домен
            };

            opsetTable.AddRow(new Markup(domainName), new Text(import.Version.ToString()));
        }

        return opsetTable;
    }



    private IRenderable InputOutputInfoAsTable(ModelProto model, bool showTitle = true)
    {
        var graph = model.Graph;

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .AddColumn("[grey]Name[/]")
            .AddColumn("[bold cyan]Type[/]")
            .AddColumn("[bold orange1]Shape[/]");

        if(showTitle)
        {
            var title = @$"MODEL [{inColor}]INPUTS[/]/[{outColor}]OUTPUTS[/]";
            table.Title($"[bold]{title}[/]");
        }

        foreach(var valueInfo in graph.Input) OutRow(true, valueInfo);
        foreach(var valueInfo in graph.Output) OutRow(false, valueInfo);

        return table;

        void OutRow(bool input, ValueInfoProto valueInfo)
        {
            (string typeStr, string shapeStr) = ValueInfoProtoToStr(valueInfo);

            // Добавляем строку в таблицу
            table.AddRow(
                $"[{(input ? inColor : outColor)}]{valueInfo.Name}[/]",
                typeStr,
                shapeStr
            );

        }
    }

    (string typeStr, string shapeStr) ValueInfoProtoToStr(ValueInfoProto valueInfo)
    {
        string typeStr = "Unknown";
        string shapeStr = "";

        // Смотрим, какой именно тип данных лежит в TypeProto
        switch(valueInfo.Type.ValueCase)
        {
            case TypeProto.ValueOneofCase.TensorType:
                {
                    // Старая логика для тензоров
                    var tensorType = valueInfo.Type.TensorType;
                    typeStr = GetDataTypeName(tensorType.ElemType);
                    shapeStr = FormatShape(tensorType.Shape);
                }
                break;

            case TypeProto.ValueOneofCase.SequenceType:
                {
                    // Логика для Sequence (например, Sequence<Tensor<float>>)
                    var seqType = valueInfo.Type.SequenceType;

                    // Обычно внутри Sequence лежит Tensor, проверяем это
                    if(seqType.ElemType.ValueCase == TypeProto.ValueOneofCase.TensorType)
                    {
                        var innerTensor = seqType.ElemType.TensorType;
                        // Пишем Seq<FP32>
                        typeStr = $"Seq<{GetDataTypeName(innerTensor.ElemType).Trim()}>";
                        // Показываем форму вложенного тензора, если есть
                        shapeStr = FormatShape(innerTensor.Shape);
                    }
                    else if(seqType.ElemType.ValueCase == TypeProto.ValueOneofCase.MapType)
                    {
                        typeStr = "Seq<Map>";
                        shapeStr = "[grey]Sequence of Maps[/]";
                    }
                    else
                    {
                        typeStr = "Sequence";
                        shapeStr = "[grey]Generic Sequence[/]";
                    }
                }
                break;

            case TypeProto.ValueOneofCase.MapType:
                {
                    var mapType = valueInfo.Type.MapType;
                    // KeyType всегда int (enum), ValueType — TypeProto
                    // Пример: Map<I64, FP32>
                    // Для значения рекурсивно проверяем, тензор ли это
                    string valTypeStr = mapType.ValueType.ValueCase == TypeProto.ValueOneofCase.TensorType
                        ? GetDataTypeName(mapType.ValueType.TensorType.ElemType).Trim()
                        : "Complex";

                    typeStr = $"Map<{GetDataTypeName(mapType.KeyType).Trim()},{valTypeStr}>";
                    shapeStr = "[grey]Key-Value Map[/]";
                }
                break;

            case TypeProto.ValueOneofCase.None:
            default:
                typeStr = "[red]Unknown[/]";
                shapeStr = "[grey]?[/]";
                break;
        }

        return (typeStr, shapeStr);
    }


    // Вынес форматирование шейпа в отдельную локальную функцию (или метод класса)
    // чтобы переиспользовать для Tensor и Sequence<Tensor>
    string FormatShape(TensorShapeProto shape)
    {
        if(shape == null) return "[grey]Scalar/Null[/]";

        var shapeParts = shape.Dim.Select(d =>
            d.HasDimValue
                ? $"[{dimValueColor}]{d.DimValue}[/]"
                : $"[{dimParamColor}]{d.DimParam ?? "?"}[/]");

        return $"[[{string.Join(" x ", shapeParts)}]]";
    }

    #endregion


    #region helpers

    // Маппинг индексов ONNX в понятные названия
    private string GetDataTypeName(int typeIndex) => typeIndex switch
    {
        //1 => "FLOAT  (Float32)",
        //2 => "UINT8  (Byte)   ",
        //3 => "INT8   (SByte)  ",
        //10 => "FLOAT16(FP16)   ",
        //11 => "DOUBLE (Float64)",
        //_ => $"Undef  ({typeIndex,7})"
        1 => "FP32",   // Float
        2 => "U8  ",     // Uint8
        3 => "I8  ",     // Int8
        6 => "I32 ",    // Int32
        7 => "I64 ",    // Int64
        9 => "BOOL",   // Boolean
        10 => "FP16",   // Float16
        11 => "FP64",   // Double
        _ => $"?({typeIndex})"
    };
    #endregion




    // Вспомогательный метод для парсинга Shape (C# 14)
    static string GetShapeString(ValueInfoProto info)
    {
        var shape = info.Type.TensorType.Shape;
        if(shape == null || shape.Dim.Count == 0) return "[grey]Unknown[/]";

        // Используем Collection Expressions и LINQ
        var dims = shape.Dim.Select(d => d.DimValue > 0 ? d.DimValue.ToString() : (string.IsNullOrEmpty(d.DimParam) ? "?" : d.DimParam));
        return $"[[{string.Join(", ", dims)}]]";
    }
}
