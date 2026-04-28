using NeuroModFlowNet.ONNX.Tools.ONNX;
using Spectre.Console;
using Spectre.Console.Cli;

namespace NeuroModFlowNet.ONNX.Tools.CLI;

public class ViewCommand : Command<ViewSettings>
{
    protected override int Execute(CommandContext context, ViewSettings settings, CancellationToken cancellationToken)
    {
        if(!File.Exists(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Model file not found at: [yellow]{settings.ModelPath}[/]");
            return 1;
        }

        // Создаем анализатор
        var analyzer = new OnnxModelAnalyzer_Spectre_Console
        {
            // Если указан флаг --all, ставим -1 (показать всё)
            TopN = settings.ShowAll ? -1 : settings.TopN,
            BottomN = settings.ShowAll ? -1 : settings.BottomN,

            // Преобразуем строку фильтра "IOV" в HashSet<char>
            IOVT = new HashSet<char>(settings.Filter.ToUpper().ToCharArray()),

            ShowNodeIONames = settings.ShowNames
        };

        try
        {
            // Запускаем инспекцию
            analyzer.InspectModel(settings.ModelPath);
        }
        catch(Exception ex)
        {
            AnsiConsole.WriteException(ex, ExceptionFormats.ShortenEverything);
            return 1;
        }

        return 0;
    }



    private string FormatShape(ValueInfoProto info)
    {
        var dims = info.Type.TensorType.Shape.Dim.Select(d =>
            d.ValueCase == TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue ? d.DimValue.ToString() : "?");
        return string.Join("x", dims);
    }
}
