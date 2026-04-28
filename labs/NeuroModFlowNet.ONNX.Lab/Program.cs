using NeuroModFlowNet.ONNX;
using OnnxTestLoader;

namespace NeuroModFlowNet.ONNX.Lab;

internal class Program
{
    private const string ModelsRootPathEnvName = "MODELS_ROOT_PATH";
    private const string DefaultModelsRootPath = @"C:\Models\det-to-obb";

    static void Main(string[] args)
    {
        OnnxRuntimePathHelper.InitFromConfig();
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        if(string.IsNullOrEmpty(Environment.GetEnvironmentVariable(ModelsRootPathEnvName)))
            Environment.SetEnvironmentVariable(ModelsRootPathEnvName, DefaultModelsRootPath);

        using RealTimeView2 realTimeView = new();
        realTimeView.Run();
    }
}
