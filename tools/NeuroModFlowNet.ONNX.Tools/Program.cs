using NeuroModFlowNet.ONNX.Tools.CLI;
using NeuroModFlowNet.ONNX.Tools.Modify;
using Spectre.Console;
using Spectre.Console.Cli;
using System.Globalization;

namespace NeuroModFlowNet.ONNX.Tools;

internal class Program
{

    static int Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.GetCultureInfo("en");
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo("en");

        var version = typeof(Program).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .Cast<System.Reflection.AssemblyInformationalVersionAttribute>()
            .FirstOrDefault()?.InformationalVersion ?? "26.0000.0000";

        AnsiConsole.MarkupLine("[bold green]NeuroModFlowNet.ONNX.Tools[/] [grey]v{0}[/]", version);
        AnsiConsole.WriteLine();

        var app = new CommandApp<ViewCommand>();

        app.Configure(config =>
        {
            config.SetApplicationName("nmf-onnx-tool");

            config.AddCommand<ViewCommand>("view")
                .WithDescription("View model structure");

            // 2. Inject: Flatter and faster (Action + Target)
            // We use "inject" as the primary verb, but add "head" as an alias
            config.AddCommand<InjectCommand>("inject")
                  .WithAlias("head") // Allows: nmf-onnx head ByteBGR ...
                  .WithDescription(OnnxHeadRegistry.GetDetailedHelp());

            //// 3. Добавляем ветку модификации
            //config.AddBranch("modify", modify =>
            //{
            //    modify.SetDescription("Modify the ONNX model structure");

            //    modify.AddBranch("add", add =>
            //    {
            //        add.SetDescription("Add new elements to the model graph");

            //        var headCommand = add.AddCommand<AddHeadCommand>("head")
            //                 .WithDescription(OnnxHeadRegistry.GetDetailedHelp());

            //        //// Автоматически добавляем все примеры из реестра
            //        //foreach(var example in OnnxHeadRegistry.GetExamples())
            //        //{
            //        //    headCommand.WithExample(example);
            //        //}
            //    });
            //});

            // 3. Команда регистрации (register)
            // Мы добавляем её на верхний уровень, чтобы было "nmf-onnx register"
            config.AddCommand<RegisterCommand>("register")
                .WithDescription("Register the application in the system (file associations and aliases)")
                .WithExample(["register"])
                .WithExample(["register", "--alias", "nmf-onnx"])
                .WithExample(["register", "--unregister"]);


        });

#if WINDOWS
        if(ConsoleHelper.IsLaunchedFromExplorer())
        {
            // Оптимальный размер для широких таблиц ONNX графа
            OptimizeForSpectre(160, 50);
        }
#endif

        int result = app.Run(args);

#if WINDOWS
        // System check: Pause if the console will disappear
        if(ConsoleHelper.IsLaunchedFromExplorer())
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Execution finished. Press any key to exit...[/]");
            Console.ReadKey(true);
        }
#endif

        return result;
    }


    public static void OptimizeForSpectre(int width, int height)
    {
        if(!OperatingSystem.IsWindows()) return;

        try
        {
            // 1. Устанавливаем размер буфера (память консоли)
            // Делаем ширину буфера равной желаемой, чтобы не было горизонтального скролла
            if(Console.BufferWidth < width)
            {
                Console.SetBufferSize(width, Math.Max(height, 2000));
            }

            // 2. Устанавливаем размер окна
            // Проверяем LargestWindowWidth, чтобы не вылететь с Exception на маленьких экранах
            int targetWidth = Math.Min(width, Console.LargestWindowWidth);
            int targetHeight = Math.Min(height, Console.LargestWindowHeight);

            Console.SetWindowSize(targetWidth, targetHeight);

            // 3. Сбрасываем кэш Spectre.Console
            // Это заставит Spectre пересчитать ширину экрана при следующем выводе
        }
        catch
        {
            // На Windows Terminal (Win11) SetWindowSize может кинуть исключение.
            // В этом случае мы просто игнорируем его, буфер уже настроен.
        }
    }

}
