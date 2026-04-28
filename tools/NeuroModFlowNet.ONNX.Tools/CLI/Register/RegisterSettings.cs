using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;


[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
public sealed class RegisterSettings : CommandSettings
{
    [CommandOption("-u|--unregister")]
    [Description("Remove application registration and file associations from the system")]
    public bool Unregister { get; set; }

    [CommandOption("-f|--force")]
    [Description("Overwrite existing associations")]
    public bool Force { get; set; }

    [CommandOption("--alias <NAME>")]
    [Description("Command name to use in the terminal (Linux symlink)")]
    [DefaultValue("nmf-onnx")]
    public string Alias { get; set; } = "nmf-onnx";

    // Валидация для системного порядка
    public override ValidationResult Validate()
    {
        if(string.IsNullOrWhiteSpace(Alias))
            return ValidationResult.Error("Command alias cannot be empty.");

        // Проверка на недопустимые символы в имени команды для Linux
        if(Alias.Any(c => char.IsWhiteSpace(c) || Path.GetInvalidFileNameChars().Contains(c)))
            return ValidationResult.Error("Command alias contains invalid characters.");

        return ValidationResult.Success();
    }
}
