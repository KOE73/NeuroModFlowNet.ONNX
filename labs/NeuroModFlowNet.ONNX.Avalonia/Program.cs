using Avalonia;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.ONNX.Avalonia;

/// <summary>
/// EN: Starts the Avalonia realtime lab application.
/// RU: Запускает Avalonia-приложение realtime lab.
/// </summary>
/// <remarks>
/// EN: Initializes ONNX Runtime native paths, sets the default models root when it is missing, and starts the desktop UI.
/// RU: Инициализирует native-пути ONNX Runtime, задает корневую папку моделей по умолчанию и запускает desktop UI.
/// </remarks>
internal static class Program
{
    private const string ModelsRootPathEnvName = "MODELS_ROOT_PATH";
    private const string DefaultModelsRootPath = @"C:\Models\det-to-obb";

    [STAThread]
    public static void Main(string[] args)
    {
        OnnxRuntimePathHelper.InitFromConfig();

        if(string.IsNullOrEmpty(Environment.GetEnvironmentVariable(ModelsRootPathEnvName)))
            Environment.SetEnvironmentVariable(ModelsRootPathEnvName, DefaultModelsRootPath);

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();
}
