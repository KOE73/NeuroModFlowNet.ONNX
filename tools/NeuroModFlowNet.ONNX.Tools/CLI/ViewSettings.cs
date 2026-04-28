using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;

[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
public class ViewSettings : CommandSettings
{
    [CommandArgument(0, "<ONNX_FILE>")]
    [Description("Путь к файлу модели .onnx")]
    public string ModelPath { get; set; } = null!;

    [CommandOption("-t|--top <VALUE>")]
    [Description("Показать первые N узлов графа")]
    [DefaultValue(5)]
    public int TopN { get; set; }

    [CommandOption("-b|--bottom <VALUE>")]
    [Description("Показать последние N узлов графа")]
    [DefaultValue(5)]
    public int BottomN { get; set; }

    [CommandOption("-a|--all")]
    [Description("Показать все узлы графа (игнорирует top/bottom)")]
    public bool ShowAll { get; set; }

    [CommandOption("-f|--filter <IOVT>")]
    [Description("Фильтр элементов: I (Input), O (Output), V (Value), T (Tensor/Weights), A (Attributes)")]
    [DefaultValue("IOVTA")]
    public string Filter { get; set; } = "IOVTA";

    [CommandOption("--names")]
    [Description("Показывать полные имена входов/выходов вместо сокращений I/O/V/T")]
    public bool ShowNames { get; set; }
}
