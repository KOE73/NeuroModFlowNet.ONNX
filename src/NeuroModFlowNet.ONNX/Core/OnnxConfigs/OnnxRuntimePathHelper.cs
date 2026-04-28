using System;
using System.IO;
using System.Linq;
using System.Configuration;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Helper for configuring system environment paths for ONNX dependencies (CUDA, cuDNN, TensorRT) via App.config or AppSettings.
/// RU: Помощник для настройки путей системного окружения для зависимостей ONNX (CUDA, cuDNN, TensorRT) через App.config.
/// </summary>
public static class OnnxRuntimePathHelper
{
    private const string KeyCuda = "CudaBinPath";
    private const string KeyCudnn = "CudnnBinPath";
    private const string KeyTrt = "TrtLibPath";

    private static bool _initialized = false;
    private static readonly object _lock = new object();

    /// <summary>
    /// EN: Configures the process PATH using settings from App.config.
    /// RU: Настраивает PATH процесса, используя настройки из App.config.
    /// </summary>
    public static void InitFromConfig()
    {
        if (_initialized) return;

        lock (_lock)
        {
            if (_initialized) return;

            try
            {
                string? cuda = ConfigurationManager.AppSettings[KeyCuda];
                string? cudnn = ConfigurationManager.AppSettings[KeyCudnn];
                string? trt = ConfigurationManager.AppSettings[KeyTrt];

                if (!string.IsNullOrWhiteSpace(cuda)) AddToSystemPath(cuda);
                if (!string.IsNullOrWhiteSpace(cudnn)) AddToSystemPath(cudnn);
                if (!string.IsNullOrWhiteSpace(trt)) AddToSystemPath(trt);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to load paths from App.config: {ex.Message}");
            }

            _initialized = true;
        }
    }

    /// <summary>
    /// EN: Adds a directory to the beginning of the current process PATH if it exists and is not already present.
    /// RU: Добавляет директорию в начало PATH текущего процесса, если она существует и еще не добавлена.
    /// </summary>
    public static void AddToSystemPath(string? dirPath)
    {
        if (string.IsNullOrWhiteSpace(dirPath) || !Directory.Exists(dirPath)) return;

        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";

        // Check if path already exists (case-insensitive)
        if (!currentPath.Split(';').Any(p => p.Trim().Equals(dirPath.Trim(), StringComparison.OrdinalIgnoreCase)))
        {
            // Add to the beginning to ensure these DLLs are found first
            string newPath = $"{dirPath};{currentPath}";
            Environment.SetEnvironmentVariable("PATH", newPath, EnvironmentVariableTarget.Process);
        }
    }
}
