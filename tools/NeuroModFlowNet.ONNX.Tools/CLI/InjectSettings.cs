using NeuroModFlowNet.ONNX.Tools.Modify;
using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;

[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
public sealed class InjectSettings : CommandSettings
{
    [CommandArgument(0, "<HEAD_TYPE>")]
    [Description("Type of the head to inject: 'ByteBGR_FP16', 'ByteBGR_FP32' or 'SequenceByteBGR'")]
    public string HeadType { get; set; } = null!;

    [CommandArgument(1, "<MODEL_PATH>")]
    [Description("Path to the source ONNX model file")]
    public string ModelPath { get; set; } = null!;

    [CommandOption("-o|--output <PATH>")]
    [Description("Output path for the modified model (default: original_path + extra_name)")]
    public string? OutputPath { get; set; }

    [CommandOption("-e|--extra <NAME>")]
    [Description("Override the default extra name suffix (e.g., '_custom')")]
    public string? ExtraName { get; set; }

    public override ValidationResult Validate()
    {
        if(!File.Exists(ModelPath))
            return ValidationResult.Error($"Source model file not found: {ModelPath}");

        var available = OnnxHeadRegistry.GetAvailableNames().ToList();
        if(!available.Contains(HeadType, StringComparer.OrdinalIgnoreCase))
        {
            return ValidationResult.Error(
                $"Unsupported head type '{HeadType}'.\n" +
                $"Available: [yellow]{string.Join(", ", available)}[/]");
        }

        return ValidationResult.Success();
    }
}
