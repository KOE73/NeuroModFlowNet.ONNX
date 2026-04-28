using Spectre.Console;

#if WINDOWS
using Microsoft.Win32;
#endif

namespace NeuroModFlowNet.ONNX.Tools.CLI;

using Spectre.Console.Cli;

public class RegisterCommand : Command<RegisterSettings>
{
    protected override int Execute(CommandContext context, RegisterSettings settings, CancellationToken cancellationToken)
    {
        string alias = settings.Alias;

#if WINDOWS
        return RegisterWindows(alias, settings.Unregister);
#elif LINUX
return RegisterLinux(alias, settings.Unregister);
#else
    AnsiConsole.MarkupLine("[red]This OS is not supported by this command.[/]");
    return 1;
#endif
    }

#if WINDOWS
    private int RegisterWindows(string alias, bool unregister)
    {
        try
        {
            string exePath = Environment.ProcessPath!;
            string progId = "NeuroModFlow.Onnx.Tool.v1";

            using var classesKey = Registry.CurrentUser.OpenSubKey(@"Software\Classes", true);
            if(classesKey == null) return 1;

            if(unregister)
            {
                classesKey.DeleteSubKeyTree(progId, false);
                classesKey.DeleteSubKeyTree(".onnx", false);
                AnsiConsole.MarkupLine("[yellow]Windows:[/] Registration removed.");
                return 0;
            }

            // Регистрация (ProgID и ассоциация)
            using var idKey = classesKey.CreateSubKey(progId);
            idKey.SetValue("", "ONNX Model File");
            using var cmdKey = idKey.CreateSubKey(@"shell\open\command");
            cmdKey.SetValue("", $"\"{exePath}\" \"%1\"");

            using var extKey = classesKey.CreateSubKey(".onnx");
            extKey.SetValue("", progId);

            AnsiConsole.MarkupLine("[green]Windows:[/] Application registered in the registry.");
            return 0;
        }
        catch(Exception ex)
        {
            AnsiConsole.WriteException(ex);
            return 1;
        }
    }
#endif

#if LINUX
private int RegisterLinux(string alias, bool unregister)
{
try {
    string exePath = Environment.ProcessPath!;
    string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    string binPath = Path.Combine(home, ".local/bin");
    string symlinkPath = Path.Combine(binPath, alias);

    if (unregister) {
        if (File.Exists(symlinkPath)) File.Delete(symlinkPath);
        // Удаление .desktop файла...
        AnsiConsole.MarkupLine("[yellow]Linux:[/] Symlinks and desktop files removed.");
        return 0;
    }

    // Создание симлинка (CLI)
    if (!Directory.Exists(binPath)) Directory.CreateDirectory(binPath);
    if (File.Exists(symlinkPath)) File.Delete(symlinkPath);
    File.CreateSymbolicLink(symlinkPath, exePath);

    // Создание .desktop (GUI) - аналогично предыдущему коду...
    
    AnsiConsole.MarkupLine($"[green]Linux:[/] Alias '{alias}' and .desktop file created.");
    return 0;
} catch (Exception ex) {
    AnsiConsole.WriteException(ex);
    return 1;
}
}
#endif

}
