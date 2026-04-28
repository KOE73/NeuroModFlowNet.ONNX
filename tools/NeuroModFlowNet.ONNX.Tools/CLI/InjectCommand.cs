using NeuroModFlowNet.ONNX.Tools.Modify;
using Spectre.Console;
using Spectre.Console.Cli;
using System.Diagnostics.CodeAnalysis;

namespace NeuroModFlowNet.ONNX.Tools.CLI;

public sealed class InjectCommand : Command<InjectSettings>
{
    protected override int Execute([NotNull] CommandContext context, [NotNull] InjectSettings settings, CancellationToken ct)
    {
        if(ct.IsCancellationRequested) return -1;
        Exception? injectionException = null;

        try
        {
            // 1. Select the appropriate injector based on user input
            OnnxModelModifier injector = OnnxHeadRegistry.Create(settings.HeadType);

        if(!string.IsNullOrEmpty(settings.ExtraName))
        {
            injector.ExtraName = settings.ExtraName;
        }


        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start($"Injecting [bold yellow]{settings.HeadType}[/] head into [white]{Path.GetFileName(settings.ModelPath)}[/]...",
                ctx =>
                {
                    try
                    {
                        // 2. Load and modify
                        // If OutputPath is specified, we handle saving manually or pass it to the injector
                        if(string.IsNullOrEmpty(settings.OutputPath))
                        {
                            // Standard behavior: save to original_dir + extra_name
                            injector.Inject(settings.ModelPath);
                        }
                        else
                        {
                            // Custom output path behavior
                            var model = injector.LoadModel(settings.ModelPath);
                            injector.Inject(model);

                            using var output = File.Create(settings.OutputPath);
                            model.WriteTo(output);
                            AnsiConsole.MarkupLine($"[green]SUCCESS:[/] Model saved to: {settings.OutputPath}");
                        }
                    }
                    catch(Exception ex)
                    {
                        injectionException = ex;
                        AnsiConsole.WriteException(ex);
                    }
                }
            );

            if(injectionException is not null)
            {
                return 1;
            }
        }
        catch(Exception ex)
        {
            AnsiConsole.WriteException(ex);
            return 1;
        }

        return 0;
    }
}
