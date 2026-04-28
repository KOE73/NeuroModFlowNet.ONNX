using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;

[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
public class ViewSettings : CommandSettings
{
    [CommandArgument(0, "<ONNX_FILE>")]
    [Description("Path to the .onnx model file")]
    public string ModelPath { get; set; } = null!;

    [CommandOption("-t|--top <VALUE>")]
    [Description("Show the first N graph nodes")]
    [DefaultValue(5)]
    public int TopN { get; set; }

    [CommandOption("-b|--bottom <VALUE>")]
    [Description("Show the last N graph nodes")]
    [DefaultValue(5)]
    public int BottomN { get; set; }

    [CommandOption("-a|--all")]
    [Description("Show all graph nodes (ignores top/bottom)")]
    public bool ShowAll { get; set; }

    [CommandOption("-f|--filter <IOVT>")]
    [Description("Element filter: I (Input), O (Output), V (Value), T (Tensor/Weights), A (Attributes)")]
    [DefaultValue("IOVTA")]
    public string Filter { get; set; } = "IOVTA";

    [CommandOption("--names")]
    [Description("Show full input/output names instead of short I/O/V/T labels")]
    public bool ShowNames { get; set; }
}
