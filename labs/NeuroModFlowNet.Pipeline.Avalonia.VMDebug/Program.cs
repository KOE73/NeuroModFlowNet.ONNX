using Avalonia;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug;

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
