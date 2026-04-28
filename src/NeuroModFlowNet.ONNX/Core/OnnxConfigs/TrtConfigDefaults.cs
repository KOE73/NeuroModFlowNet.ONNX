namespace NeuroModFlowNet.ONNX;

public static class TrtConfigDefaults
{
    /// <summary>
    /// Gets or sets a custom path for the TensorRT cache. 
    /// If set, this overrides the environment variable NMFN_ONNX_TRT_CACHE_PATH and the OS default.
    /// </summary>
    public static string? CustomEngineCachePath { get; set; }

    /// <summary>
    /// Resolves the TensorRT engine cache path with the following priority:
    /// 1. CustomEngineCachePath (if set in code)
    /// 2. NMFN_ONNX_TRT_CACHE_PATH (environment variable)
    /// 3. Platform-specific local application data path (LocalApplicationData/NeuroModFlowNet/TRT_Cache)
    /// </summary>
    public static string GetEngineCachePath()
    {
        if(!string.IsNullOrWhiteSpace(CustomEngineCachePath))
        {
            EnsureDirectoryExists(CustomEngineCachePath);
            return CustomEngineCachePath;
        }

        string? envPath = Environment.GetEnvironmentVariable("NMFN_ONNX_TRT_CACHE_PATH");
        if(!string.IsNullOrWhiteSpace(envPath))
        {
            EnsureDirectoryExists(envPath);
            return envPath;
        }

        string defaultPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "NeuroModFlowNet",
            "TRT_Cache"
        );
        EnsureDirectoryExists(defaultPath);
        return defaultPath;
    }

    static void EnsureDirectoryExists(string path)
    {
            Directory.CreateDirectory(path);
    }
}
