using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;


[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
public sealed class RegisterSettings : CommandSettings
{
    [CommandOption("-u|--unregister")]
    [Description("Удалить регистрацию программы и ассоциации файлов из системы")]
    public bool Unregister { get; set; }

    [CommandOption("-f|--force")]
    [Description("Принудительно перезаписать существующие ассоциации")]
    public bool Force { get; set; }

    [CommandOption("--alias <NAME>")]
    [Description("Имя команды для вызова в терминале (актуально для Linux symlink)")]
    [DefaultValue("nmf-onnx")]
    public string Alias { get; set; } = "nmf-onnx";

    // Валидация для системного порядка
    public override ValidationResult Validate()
    {
        if(string.IsNullOrWhiteSpace(Alias))
            return ValidationResult.Error("Имя команды (alias) не может быть пустым");

        // Проверка на недопустимые символы в имени команды для Linux
        if(Alias.Any(c => char.IsWhiteSpace(c) || Path.GetInvalidFileNameChars().Contains(c)))
            return ValidationResult.Error("Имя команды содержит недопустимые символы");

        return ValidationResult.Success();
    }
}
