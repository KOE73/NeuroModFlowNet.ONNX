using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Lab.Algorithms;

internal sealed class AlgorithmLog
{
    private readonly List<string> _lines = [];

    public void Add(string message) => _lines.Add(message);

    public void Write(string title)
    {
        if(_lines.Count == 0) return;

        Table table = new Table()
            .Title($"{title} log")
            .AddColumn("Message");

        foreach(string line in _lines)
            table.AddRow(FormatLine(line));

        AnsiConsole.Write(table);
    }

    private static string FormatLine(string line)
    {
        if(line.All(character => character == '-'))
            return "[grey]" + Markup.Escape(line) + "[/]";

        if(line.Contains(" OK", StringComparison.Ordinal))
            return "[green]" + Markup.Escape(line) + "[/]";

        if(line.Contains("Error", StringComparison.Ordinal))
            return "[red]" + Markup.Escape(line) + "[/]";

        if(line.StartsWith("[AutoTune]", StringComparison.Ordinal))
            return "[bold cyan]" + Markup.Escape(line) + "[/]";

        return Markup.Escape(line);
    }
}
