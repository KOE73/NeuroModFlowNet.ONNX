using Microsoft.ML.OnnxRuntime;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace NeuroModFlowNet.ONNX.Diagnostics;

public static class OnnxRuntimeContextDiagnosticsExtensions
{
    public static TableBorder TensorTableBorder = TableBorder.Minimalist;
    private static readonly Style InputStyle = new(foreground: Color.Green);
    private static readonly Style OutputStyle = new(foreground: Color.Yellow);
    private static readonly Style HeaderStyle = new(foreground: Color.Yellow);


    private static readonly string[] DefaultMetadataKeys = ["names", "args", "stride"];

    public static void WriteInfo(this OnnxRuntimeContext context, bool includeMetadata = false, IReadOnlyCollection<string>? metadataKeys = null)
        => context.WriteInfo(AnsiConsole.Console, includeMetadata, metadataKeys);

    public static void WriteInfo(
        this OnnxRuntimeContext context,
        IAnsiConsole console,
        bool includeMetadata = false,
        IReadOnlyCollection<string>? metadataKeys = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(console);

        Grid root = new Grid()
            .AddColumn();

        root.AddRow(CreateHeaderGrid(context));
        root.AddRow(CreateTensorTable(context));

        if(includeMetadata && context.Session.ModelMetadata.CustomMetadataMap.Count > 0)
        {
            IReadOnlyDictionary<string, string> metadata = SelectMetadata(context.Session.ModelMetadata.CustomMetadataMap, metadataKeys);
            if(metadata.Count > 0)
                root.AddRow(CreateMetadataTable(metadata));
        }

        Panel panel = new Panel(root)
            .Header(CreateHeader(context))
            .Border(BoxBorder.Rounded)
            .Padding(1, 0);

        console.Write(panel);
    }

    private static Grid CreateHeaderGrid(OnnxRuntimeContext context)
    {
        Grid grid = new Grid()
            .AddColumn(new GridColumn().NoWrap().PadRight(2))
            .AddColumn();

        grid.AddRow("[grey]Path[/]", Markup.Escape(context.ModelPath));

        return grid;
    }

    //ONNX Runtime Context
    private static string CreateHeader(OnnxRuntimeContext context) =>
        $" ONNX Runtime Context " +
        $"[yellow]{Markup.Escape(context.InferenceBackend.ToString())}[/] " +
        $"[bold cyan] {Markup.Escape(Path.GetFileName(context.ModelPath))} [/] "
        ;
    private static Table CreateTensorTable(OnnxRuntimeContext context)
    {
        Table table = new Table()
            .Border(TensorTableBorder)
            .AddColumn(new TableColumn(new Text("Dir", HeaderStyle)))
            .AddColumn(new TableColumn(new Text("Name", HeaderStyle)))
            .AddColumn(new TableColumn(new Text("Type", HeaderStyle)))
            .AddColumn(new TableColumn(new Text("Shape", HeaderStyle)));

        foreach((string name, NodeMetadata node) in context.Session.InputMetadata)
            table.AddRow(
                new Text("IN", InputStyle),
                new Text(name, InputStyle),
                new Text(node.ElementDataType.ToString(), InputStyle),
                new Text(FormatShape(node.Dimensions), InputStyle));

        foreach((string name, NodeMetadata node) in context.Session.OutputMetadata)
            table.AddRow(
                new Text("OUT", OutputStyle),
                new Text(name, OutputStyle),
                new Text(node.ElementDataType.ToString(), OutputStyle),
                new Text(FormatShape(node.Dimensions), OutputStyle));

        return table;
    }

    private static Table CreateMetadataTable(IReadOnlyDictionary<string, string> metadata)
    {
        Table table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("Key")
            .AddColumn("Value");

        foreach((string key, string value) in metadata)
            table.AddRow(Markup.Escape(key), Markup.Escape(value));

        return table;
    }

    private static IReadOnlyDictionary<string, string> SelectMetadata(
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyCollection<string>? metadataKeys)
    {
        if(metadataKeys is { Count: 0 })
            return metadata;

        string[] keys = metadataKeys?.ToArray() ?? DefaultMetadataKeys;
        Dictionary<string, string> result = new(keys.Length);

        foreach(string key in keys)
        {
            if(metadata.TryGetValue(key, out string? value))
                result[key] = value;
        }

        return result;
    }

    private static string FormatShape(IEnumerable<int> shape) =>
        "[" + string.Join(" x ", shape.Select(dimension => dimension < 0 ? "?" : dimension.ToString())) + "]";

}
